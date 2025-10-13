interface VectorBase {
  id: string;
  metadata?: Record<string, any>;
}

interface VectorInternal extends VectorBase {
  values: Float32Array;
  magnitude: number;
}

export interface Vector extends VectorBase {
  values: number[];
}

export interface InsertArgs {
  values: number[];
  metadata?: Record<string, any>;
}

export interface SearchArgs {
  query: number[];
  topK: number;
}

export interface SearchResult {
  vector: Vector;
  score: number;
}

export class Dot {
  static readonly STORE_NAME = "vectors";

  private _db: IDBDatabase | null;
  private vectors: Map<string, VectorInternal>;
  private readonly dimension: number;

  constructor(dimension: number) {
    if (dimension <= 0) {
      throw new Error("Dimension must be a positive number.");
    }

    this._db = null;
    this.vectors = new Map<string, VectorInternal>();
    this.dimension = dimension;
  }

  async connect(dbName: string): Promise<{ db: IDBDatabase; isNew: boolean }> {
    return new Promise((resolve, reject) => {
      const dbReq = indexedDB.open(dbName, 1);
      let isNew = false;

      dbReq.onerror = (e) => {
        reject((e.target as IDBOpenDBRequest).error);
      };

      dbReq.onupgradeneeded = (e) => {
        const db = (e.target as IDBOpenDBRequest).result;

        // Create an object store for vectors
        if (!db.objectStoreNames.contains(Dot.STORE_NAME)) {
          db.createObjectStore(Dot.STORE_NAME, { keyPath: "id" });
        }

        isNew = true;
      };

      dbReq.onsuccess = async (event) => {
        try {
          this._db = (event.target as IDBOpenDBRequest).result;

          const vectors = await this.dbGetAllVectors();
          if (vectors.length > 0) {
            // Verify that the dimension of vectors in the DB matches
            // the dimension of this  instance.
            const dbDimension = vectors[0].values.length;
            if (dbDimension !== this.dimension) {
              reject(
                new Error(
                  `Database vector dimension mismatch. DB is dimension ${dbDimension}, instance is configured for ${this.dimension}.`
                )
              );
              return;
            }
          }

          for (const vector of vectors) {
            this.vectors.set(vector.id, vector);
          }

          resolve({
            db: this._db,
            isNew: isNew,
          });
        } catch (e) {
          reject(e);
        }
      };
    });
  }

  get db(): IDBDatabase {
    if (this._db === null) {
      throw new Error("Database not connected");
    }

    return this._db;
  }

  dot(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) {
      throw new Error("Vectors must be of the same length");
    }

    let dotProduct = 0;
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
    }

    return dotProduct;
  }

  magnitude(vector: Float32Array): number {
    return Math.sqrt(this.dot(vector, vector));
  }

  cosineSimilarity(a: VectorInternal, b: VectorInternal): number {
    return this.dot(a.values, b.values) / (a.magnitude * b.magnitude);
  }

  search({ query, topK }: SearchArgs): SearchResult[] {
    if (query.length !== this.dimension) {
      throw new Error(`Query vector dimension mismatch. Expected ${this.dimension}, got ${query.length}`);
    }

    const results: { vector: VectorInternal; score: number }[] = [];

    const queryArray = new Float32Array(query);

    // create a query vector for optimized cosine similarity calculation
    const queryVector: VectorInternal = {
      id: "temp-query", // not really needed since we aren't storing this vector
      values: queryArray,
      magnitude: this.magnitude(queryArray),
    };

    // O(N) search (pretty bad)
    for (const vector of this.vectors.values()) {
      const score = this.cosineSimilarity(queryVector, vector);
      results.push({ vector: vector, score: score });
    }

    // sort by score (again, pretty bad)
    results.sort((a, b) => b.score - a.score);

    // return top K results
    return results.slice(0, topK).map((res) => ({
      vector: {
        id: res.vector.id,
        values: Array.from(res.vector.values),
        metadata: res.vector.metadata,
      },
      score: res.score,
    }));
  }

  getAll(): Vector[] {
    return Array.from(this.vectors.values()).map((vector) => ({
      id: vector.id,
      values: Array.from(vector.values),
      metadata: vector.metadata,
    }));
  }

  async insert({ values, metadata }: InsertArgs): Promise<string> {
    if (values.length !== this.dimension) {
      throw new Error(`Vector dimension mismatch. Expected ${this.dimension}, got ${values.length}`);
    }

    const id = crypto.randomUUID();

    const valuesArray = new Float32Array(values);

    // create internal vector
    const vector: VectorInternal = {
      id,
      values: valuesArray,
      metadata,
      magnitude: this.magnitude(valuesArray),
    };

    // insert into DB
    await this.dbInsertVector(vector);

    // in-memory storage
    this.vectors.set(id, vector);

    return id;
  }

  async insertMany(vectors: VectorInternal[]): Promise<void> {
    await this.dbInsertManyVectors(vectors);
    for (const vector of vectors) {
      this.vectors.set(vector.id, vector);
    }
  }

  async delete(id: string): Promise<void> {
    await this.dbDeleteVector(id);
    this.vectors.delete(id);
  }

  async deleteMany(ids: string[]): Promise<void> {
    if (ids.length === 0) {
      return;
    }

    await this.dbDeleteManyVectors(ids);
    for (const id of ids) {
      this.vectors.delete(id);
    }
  }

  private dbGetAll<T>(storeName: string): Promise<T[]> {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(storeName, "readonly");
      const store = transaction.objectStore(storeName);
      const request = store.getAll();

      request.onsuccess = () => {
        resolve(request.result as T[]);
      };

      request.onerror = () => {
        reject(request.error);
      };
    });
  }

  private dbInsert<T>(storeName: string, value: T): Promise<void> {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(storeName, "readwrite");
      const store = transaction.objectStore(storeName);
      const request = store.add(value);

      request.onsuccess = () => {
        resolve();
      };

      request.onerror = () => {
        reject(request.error);
      };
    });
  }

  private dbInsertMany<T>(storeName: string, values: T[]): Promise<void> {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(storeName, "readwrite");
      const store = transaction.objectStore(storeName);

      values.forEach((value) => store.add(value));

      transaction.oncomplete = () => {
        resolve();
      };

      transaction.onerror = () => {
        reject(transaction.error);
      };
    });
  }

  private dbDelete(storeName: string, key: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(storeName, "readwrite");
      const store = transaction.objectStore(storeName);
      const request = store.delete(key);

      request.onsuccess = () => {
        resolve();
      };

      request.onerror = () => {
        reject(request.error);
      };
    });
  }

  private dbDeleteMany(storeName: string, keys: string[]): Promise<void> {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(storeName, "readwrite");
      const store = transaction.objectStore(storeName);

      keys.forEach((key) => store.delete(key));

      transaction.oncomplete = () => {
        resolve();
      };

      transaction.onerror = () => {
        reject(transaction.error);
      };
    });
  }

  private dbGetAllVectors(): Promise<VectorInternal[]> {
    return this.dbGetAll<VectorInternal>(Dot.STORE_NAME);
  }

  private dbInsertVector(vector: VectorInternal): Promise<void> {
    return this.dbInsert(Dot.STORE_NAME, vector);
  }

  private dbInsertManyVectors(vectors: VectorInternal[]): Promise<void> {
    return this.dbInsertMany(Dot.STORE_NAME, vectors);
  }

  private dbDeleteVector(id: string): Promise<void> {
    return this.dbDelete(Dot.STORE_NAME, id);
  }

  private dbDeleteManyVectors(ids: string[]): Promise<void> {
    return this.dbDeleteMany(Dot.STORE_NAME, ids);
  }
}
