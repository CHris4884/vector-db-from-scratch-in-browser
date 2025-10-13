import { useState, useEffect, useRef } from "react";
import { pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";

import { Dot } from "./dotdb";
import type { SearchResult } from "./dotdb";
import { DEFAULT_TEXT } from "./default-data";

const DEFAULT_TOP_K = 5;
const MAX_TOP_K = 50;

export default function App() {
  const [db, setDb] = useState<Dot | null>(null);
  const extractor = useRef<FeatureExtractionPipeline | null>(null);
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState("Initializing...");

  // Add document form
  const [newDoc, setNewDoc] = useState(DEFAULT_TEXT);
  const [adding, setAdding] = useState(false);
  const [isAddSectionOpen, setIsAddSectionOpen] = useState(true);

  // Search
  const [searchQuery, setSearchQuery] = useState("");
  const [topK, setTopK] = useState(DEFAULT_TOP_K);
  const [topKInput, setTopKInput] = useState(DEFAULT_TOP_K.toString());
  const [searching, setSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);

  // Documents
  const [documents, setDocuments] = useState<Array<{ id: string; text: string }>>([]);

  useEffect(() => {
    async function init() {
      try {
        setStatus("Loading embedding model...");
        const ext = await pipeline("feature-extraction", "Xenova/bge-small-en-v1.5");
        extractor.current = ext;

        setStatus("Connecting to vector database...");
        const dotDb = new Dot(384); // bge-small-en-v1.5 produces 384-dimensional embeddings
        await dotDb.connect("dotdb-demo");
        setDb(dotDb);

        // Load existing documents from the database
        const existingVectors = dotDb.getAll();
        const existingDocs = existingVectors.map((vector) => ({
          id: vector.id,
          text: vector.metadata?.text || "",
        }));
        setDocuments(existingDocs);

        // Collapse the add section if documents already exist
        if (existingDocs.length > 0) {
          setIsAddSectionOpen(false);
        }

        setStatus("Ready!");
        setLoading(false);
      } catch (error) {
        console.error("Error during initialization:", error);
        setStatus(`Error: ${error}`);
      }
    }
    init();
  }, []);

  const addDocument = async () => {
    if (!newDoc.trim() || !db || !extractor.current) return;

    setAdding(true);
    try {
      // Split by paragraphs (double newline or single newline)
      const paragraphs = newDoc
        .split(/\n\n+/)
        .map((p) => p.trim())
        .filter((p) => p.length > 0);

      const newDocs: Array<{ id: string; text: string }> = [];

      // Add each paragraph separately
      for (const paragraph of paragraphs) {
        // Generate embedding for each paragraph
        const embedding = await extractor.current(paragraph, { pooling: "mean", normalize: true });
        const values = Array.from(embedding.data) as number[];

        // Insert into vector DB
        const id = await db.insert({
          values,
          metadata: { text: paragraph },
        });

        newDocs.push({ id, text: paragraph });
      }

      setDocuments([...documents, ...newDocs]);
      setNewDoc("");
      alert(`Added ${paragraphs.length} paragraph(s) to the database!`);
    } catch (error) {
      alert(`Error adding document: ${error}`);
    }
    setAdding(false);
  };

  const performSearch = async () => {
    if (!searchQuery.trim() || !db || !extractor.current) return;

    setSearching(true);
    try {
      // Generate query embedding
      const queryPrefix = "Represent this sentence for searching relevant passages: ";
      const embedding = await extractor.current(queryPrefix + searchQuery, { pooling: "mean", normalize: true });
      const values = Array.from(embedding.data) as number[];

      // Search in vector DB
      const results = db.search({ query: values, topK });
      setSearchResults(results);
    } catch (error) {
      alert(`Error searching: ${error}`);
    }
    setSearching(false);
  };

  if (loading) {
    return (
      <div className="p-8 max-w-4xl mx-auto">
        <h1 className="text-2xl font-bold mb-4">DotDB - Vector Database from Scratch</h1>
        <p>{status}</p>
      </div>
    );
  }

  return (
    <div className="p-6 flex flex-col lg:flex-row gap-6 lg:h-screen lg:overflow-hidden">
      <section className="flex-1 flex flex-col lg:overflow-hidden">
        <div className="flex justify-between items-start mb-6 lg:flex-shrink-0">
          <h1 className="text-2xl font-bold">DotDB - Vector Database from Scratch</h1>
          <a
            href="https://github.com/biraj21/vector-db-from-scratch-in-browser"
            target="_blank"
            rel="noopener noreferrer"
            className="border px-4 py-2 text-sm hover:bg-gray-100"
          >
            GitHub
          </a>
        </div>

        {/* Search Section */}
        <div className="mb-4 border p-4 lg:flex-shrink-0">
          <h2 className="text-xl font-semibold mb-3">Search</h2>
          <div className="flex gap-2 mb-2">
            <input
              className="flex-1 border p-2"
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Enter search query..."
              disabled={searching}
              onKeyDown={(e) => e.key === "Enter" && performSearch()}
            />
            <div className="flex gap-2 items-center">
              <input
                className="border p-2 w-20"
                type="number"
                min="1"
                max={MAX_TOP_K}
                placeholder="Top K"
                value={topKInput}
                onChange={(e) => {
                  setTopKInput(e.target.value);
                  const val = parseInt(e.target.value);
                  if (!isNaN(val) && val >= 1 && val <= MAX_TOP_K) {
                    setTopK(val);
                  }
                }}
                onBlur={() => {
                  const val = parseInt(topKInput);
                  if (isNaN(val) || val < 1) {
                    setTopK(DEFAULT_TOP_K);
                    setTopKInput(DEFAULT_TOP_K.toString());
                  } else if (val > MAX_TOP_K) {
                    setTopK(MAX_TOP_K);
                    setTopKInput(MAX_TOP_K.toString());
                  } else {
                    setTopKInput(val.toString());
                  }
                }}
                disabled={searching}
              />
            </div>
          </div>
          <button
            className="border px-4 py-2 disabled:opacity-50"
            onClick={performSearch}
            disabled={searching || !searchQuery.trim()}
          >
            {searching ? "Searching..." : "Search"}
          </button>
        </div>

        {/* Search Results - Scrollable */}
        <div className="lg:flex-1 lg:overflow-y-auto">
          {searchResults.length > 0 && (
            <div className="border p-4">
              <h2 className="text-xl font-semibold mb-3">Search Results</h2>
              {searchResults.map((result, idx) => (
                <div key={result.vector.id} className="mb-3 p-3 border-l-4 bg-gray-50">
                  <div className="font-semibold">Result #{idx + 1}</div>
                  <div className="text-sm">Score: {result.score.toFixed(4)}</div>
                  <div className="mt-1">{result.vector.metadata?.text}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>

      <section className="flex-1 flex flex-col lg:overflow-y-auto">
        {/* Add Document Section */}
        <div className="mb-8 border p-4">
          <div className="flex justify-between items-center mb-3">
            <h2 className="text-xl font-semibold">Add Document</h2>
            {documents.length > 0 && (
              <button className="text-sm border px-3 py-1" onClick={() => setIsAddSectionOpen(!isAddSectionOpen)}>
                {isAddSectionOpen ? "Collapse" : "Expand"}
              </button>
            )}
          </div>

          {isAddSectionOpen && (
            <>
              <textarea
                className="w-full border p-2 mb-2"
                rows={8}
                value={newDoc}
                onChange={(e) => setNewDoc(e.target.value)}
                placeholder="Paste text here. It will be split by paragraphs (double newlines) and each paragraph will be added separately to the database..."
                disabled={adding}
              />
              <button
                className="border px-4 py-2 disabled:opacity-50"
                onClick={addDocument}
                disabled={adding || !newDoc.trim()}
              >
                {adding ? "Adding..." : "Add Document"}
              </button>
            </>
          )}

          <p className="text-sm mt-2 text-gray-600">Documents stored: {documents.length}</p>
        </div>

        {/* All Documents */}
        <div className="border p-4">
          <h2 className="text-xl font-semibold mb-3">All Documents</h2>
          {documents.length === 0 ? (
            <p className="text-gray-500">No documents yet. Add some above!</p>
          ) : (
            <div>
              {documents.map((doc, idx) => (
                <div key={doc.id} className="mb-2 p-2 border-l-2">
                  <span className="text-sm font-semibold">Doc #{idx + 1}:</span> {doc.text}
                </div>
              ))}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
