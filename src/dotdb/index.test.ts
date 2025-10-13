import { Dot, type SearchResult } from ".";

export async function runTests() {
  console.log("🧪 Starting DotDB Tests\n");

  // Test 1: Basic Connection
  console.log("Test 1: Database Connection");
  const db = new Dot(3);
  const { isNew } = await db.connect("test-vector-db");

  console.log(`✓ Connected to DB (isNew: ${isNew})\n`);

  // Test 2: Insert Vectors
  console.log("Test 2: Insert Vectors");
  const id1 = await db.insert({
    values: [1, 0, 0],
    metadata: { text: "red", category: "color" },
  });
  console.log(`✓ Inserted vector 1: ${id1}`);

  const id2 = await db.insert({
    values: [0, 1, 0],
    metadata: { text: "green", category: "color" },
  });
  console.log(`✓ Inserted vector 2: ${id2}`);

  const id3 = await db.insert({
    values: [0.9, 0.1, 0],
    metadata: { text: "reddish", category: "color" },
  });
  console.log(`✓ Inserted vector 3: ${id3}\n`);

  // Test 3: Search Functionality
  console.log("Test 3: Search for Similar Vectors");
  const queryVector = [1, 0, 0]; // Should be most similar to red
  const results: SearchResult[] = db.search({
    query: queryVector,
    topK: 2,
  });

  console.log("Search Results:");
  results.forEach((result, idx) => {
    console.log(`  ${idx + 1}. Score: ${result.score.toFixed(4)} - ${result.vector.metadata?.text}`);
  });

  // Validate results
  if (results[0].vector.metadata?.text === "red") {
    console.log("✓ Top result is correct (red)\n");
  } else {
    console.log("✗ Top result is incorrect\n");
  }

  // Test 4: Persistence Check
  console.log("Test 4: Testing Persistence");
  console.log("Creating new DB instance and reconnecting...");
  const db2 = new Dot(3);
  await db2.connect("test-vector-db");

  const persistedResults = db2.search({
    query: queryVector,
    topK: 3,
  });

  console.log(`✓ Found ${persistedResults.length} vectors after reconnection`);
  console.log("Persisted vectors:");
  persistedResults.forEach((result) => {
    console.log(`  - ${result.vector.metadata?.text} (score: ${result.score.toFixed(4)})`);
  });
  console.log();

  // Test 5: Delete Functionality
  console.log("Test 5: Delete Vector");
  await db2.delete(id2);
  console.log(`✓ Deleted vector: ${id2}`);

  const afterDeleteResults = db2.search({
    query: [0, 1, 0], // green vector
    topK: 3,
  });

  console.log(`✓ Vectors remaining: ${afterDeleteResults.length}`);
  const hasGreen = afterDeleteResults.some((r) => r.vector.metadata?.text === "green");
  if (!hasGreen) {
    console.log("✓ Green vector successfully deleted\n");
  } else {
    console.log("✗ Green vector still exists\n");
  }

  // Test 6: Edge Cases
  console.log("Test 6: Edge Cases");

  // Search with topK larger than available vectors
  const moreResults = db2.search({
    query: [1, 0, 0],
    topK: 100,
  });
  console.log(`✓ Requested topK=100, got ${moreResults.length} results (expected: 2)`);

  // Insert vector without metadata
  const id4 = await db2.insert({
    values: [0, 0, 1],
  });
  console.log(`✓ Inserted vector without metadata: ${id4}\n`);

  // Test 7: Cosine Similarity Validation
  console.log("Test 7: Cosine Similarity Validation");

  // Identical vectors should have similarity = 1
  await db2.insert({
    values: [1, 1, 1],
    metadata: { text: "test-identical" },
  });

  const identicalResults = db2.search({
    query: [1, 1, 1],
    topK: 1,
  });

  if (Math.abs(identicalResults[0].score - 1.0) < 0.0001) {
    console.log(`✓ Identical vectors have similarity ≈ 1.0 (${identicalResults[0].score.toFixed(6)})`);
  } else {
    console.log(`✗ Identical vectors similarity incorrect: ${identicalResults[0].score}`);
  }

  // Orthogonal vectors should have similarity = 0
  await db2.insert({
    values: [1, 0, 0],
    metadata: { text: "x-axis" },
  });

  await db2.insert({
    values: [0, 1, 0],
    metadata: { text: "y-axis" },
  });

  const orthogonalResults = db2.search({
    query: [1, 0, 0],
    topK: 10,
  });

  const yAxisResult = orthogonalResults.find((r) => r.vector.metadata?.text === "y-axis");
  if (yAxisResult && Math.abs(yAxisResult.score - 0.0) < 0.0001) {
    console.log(`✓ Orthogonal vectors have similarity ≈ 0.0 (${yAxisResult.score.toFixed(6)})\n`);
  } else {
    console.log(`✗ Orthogonal vectors similarity incorrect: ${yAxisResult?.score}\n`);
  }

  console.log("✅ All tests completed!");

  // Cleanup
  console.log("\n🧹 Cleaning up test database...");
  indexedDB.deleteDatabase("test-vector-db");
  console.log("✓ Test database deleted");
}

// Run tests
runTests().catch(console.error);
