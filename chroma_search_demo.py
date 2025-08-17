import chromadb
import json
from typing import List, Dict, Any

class ChromaDBSearcher:
    """
    Advanced ChromaDB searcher for Hugging Face knowledge base
    """
    
    def __init__(self, db_path: str = "./chroma_db"):
        """Initialize the ChromaDB client and connect to collections"""
        self.client = chromadb.PersistentClient(path=db_path)
        self.collections = {}
        
        # Load all available collections
        self._load_collections()
    
    def _load_collections(self):
        """Load all available collections"""
        collection_names = ["ml_documentation", "code_examples", "research_papers", "faq"]
        
        for name in collection_names:
            try:
                self.collections[name] = self.client.get_collection(name=name)
                print(f"✅ Loaded collection: {name} ({self.collections[name].count()} documents)")
            except Exception as e:
                print(f"⚠️  Failed to load collection {name}: {e}")
    
    def search_all_collections(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Search across all collections and return organized results
        """
        all_results = {}
        
        for collection_name, collection in self.collections.items():
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=n_results
                )
                all_results[collection_name] = results
            except Exception as e:
                print(f"Error searching {collection_name}: {e}")
                all_results[collection_name] = None
        
        return all_results
    
    def search_specific_collection(self, collection_name: str, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search a specific collection with detailed results
        """
        if collection_name not in self.collections:
            return {"error": f"Collection '{collection_name}' not found"}
        
        try:
            results = self.collections[collection_name].query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            return {"error": str(e)}
    
    def find_code_examples(self, task: str = None, difficulty: str = None) -> List[Dict]:
        """
        Find code examples based on task type and difficulty
        """
        if "code_examples" not in self.collections:
            return []
        
        collection = self.collections["code_examples"]
        
        # If specific filters provided, search with metadata
        if task or difficulty:
            # ChromaDB doesn't support metadata filtering in query directly,
            # so we'll get all documents and filter
            all_docs = collection.get()
            filtered_results = []
            
            for i, metadata in enumerate(all_docs['metadatas']):
                match = True
                if task and metadata.get('task', '').lower() != task.lower():
                    match = False
                if difficulty and metadata.get('difficulty', '').lower() != difficulty.lower():
                    match = False
                
                if match:
                    filtered_results.append({
                        'id': all_docs['ids'][i],
                        'document': all_docs['documents'][i],
                        'metadata': metadata
                    })
            
            return filtered_results
        else:
            # Return all code examples
            all_docs = collection.get()
            return [
                {
                    'id': all_docs['ids'][i],
                    'document': all_docs['documents'][i], 
                    'metadata': all_docs['metadatas'][i]
                }
                for i in range(len(all_docs['ids']))
            ]
    
    def get_research_papers(self, year: int = None, field: str = None) -> List[Dict]:
        """
        Get research papers with optional filtering
        """
        if "research_papers" not in self.collections:
            return []
        
        collection = self.collections["research_papers"]
        all_docs = collection.get()
        filtered_results = []
        
        for i, metadata in enumerate(all_docs['metadatas']):
            match = True
            if year and metadata.get('year') != year:
                match = False
            if field and field.lower() not in metadata.get('field', '').lower():
                match = False
            
            if match:
                filtered_results.append({
                    'id': all_docs['ids'][i],
                    'document': all_docs['documents'][i],
                    'metadata': metadata
                })
        
        return filtered_results
    
    def get_faq_by_category(self, category: str = None) -> List[Dict]:
        """
        Get FAQ items by category
        """
        if "faq" not in self.collections:
            return []
        
        collection = self.collections["faq"]
        all_docs = collection.get()
        
        if category:
            filtered_results = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if metadata.get('category', '').lower() == category.lower():
                    filtered_results.append({
                        'id': all_docs['ids'][i],
                        'document': all_docs['documents'][i],
                        'metadata': metadata
                    })
            return filtered_results
        else:
            return [
                {
                    'id': all_docs['ids'][i],
                    'document': all_docs['documents'][i],
                    'metadata': all_docs['metadatas'][i]
                }
                for i in range(len(all_docs['ids']))
            ]
    
    def semantic_search_with_context(self, query: str, collection_name: str = None, n_results: int = 3) -> Dict:
        """
        Perform semantic search and provide context about results
        """
        if collection_name:
            # Search specific collection
            if collection_name not in self.collections:
                return {"error": f"Collection '{collection_name}' not found"}
            
            results = self.search_specific_collection(collection_name, query, n_results)
            return {
                "query": query,
                "collection": collection_name,
                "results": results,
                "total_found": len(results.get('documents', [[]])[0]) if 'documents' in results else 0
            }
        else:
            # Search all collections
            all_results = self.search_all_collections(query, n_results)
            total_found = sum(
                len(results.get('documents', [[]])[0]) if results and 'documents' in results else 0
                for results in all_results.values()
            )
            
            return {
                "query": query,
                "collections": "all",
                "results": all_results,
                "total_found": total_found
            }
    
    def get_collection_stats(self) -> Dict[str, Dict]:
        """
        Get statistics about all collections
        """
        stats = {}
        
        for name, collection in self.collections.items():
            try:
                count = collection.count()
                # Get sample metadata to understand structure
                sample = collection.get(limit=1)
                sample_metadata = sample['metadatas'][0] if sample['metadatas'] else {}
                
                stats[name] = {
                    "document_count": count,
                    "sample_metadata_keys": list(sample_metadata.keys()),
                    "description": collection.metadata.get('description', 'No description available')
                }
            except Exception as e:
                stats[name] = {"error": str(e)}
        
        return stats

def demo_search_functionality():
    """
    Demonstrate various search capabilities
    """
    print("🚀 ChromaDB Advanced Search Demo")
    print("=" * 60)
    
    # Initialize searcher
    searcher = ChromaDBSearcher()
    
    # 1. Show collection statistics
    print("\n📊 Collection Statistics:")
    print("-" * 30)
    stats = searcher.get_collection_stats()
    for name, stat in stats.items():
        if 'error' not in stat:
            print(f"📚 {name}: {stat['document_count']} documents")
            print(f"   Keys: {', '.join(stat['sample_metadata_keys'])}")
            print(f"   Description: {stat['description']}")
        else:
            print(f"❌ {name}: {stat['error']}")
        print()
    
    # 2. Demonstrate semantic search
    print("\n🔍 Semantic Search Examples:")
    print("-" * 30)
    
    queries = [
        "How to fine-tune BERT for sentiment analysis?",
        "What are transformers and attention mechanisms?",
        "Code example for text generation with GPT",
        "Deploy models in production environment"
    ]
    
    for query in queries:
        print(f"\n🔎 Query: '{query}'")
        results = searcher.semantic_search_with_context(query, n_results=2)
        print(f"📋 Found {results['total_found']} results across all collections")
        
        # Show top result from each collection
        for collection, collection_results in results['results'].items():
            if collection_results and 'documents' in collection_results:
                docs = collection_results['documents'][0]
                if docs:
                    metadata = collection_results['metadatas'][0][0]
                    preview = docs[0][:100] + "..." if len(docs[0]) > 100 else docs[0]
                    print(f"   📚 {collection}: {metadata.get('type', 'document')} - {preview}")
        print("-" * 40)
    
    # 3. Demonstrate filtered searches
    print("\n🎯 Filtered Search Examples:")
    print("-" * 30)
    
    # Find beginner code examples
    print("\n🔍 Beginner Code Examples:")
    beginner_examples = searcher.find_code_examples(difficulty="beginner")
    for example in beginner_examples:
        task = example['metadata'].get('task', 'Unknown')
        print(f"   📝 {task}: {example['id']}")
    
    # Find recent research papers
    print("\n🔍 Research Papers from 2018:")
    papers_2018 = searcher.get_research_papers(year=2018)
    for paper in papers_2018:
        authors = paper['metadata'].get('authors', 'Unknown')
        print(f"   📄 {paper['id']}: {authors}")
    
    # Find FAQ by category
    print("\n🔍 General FAQ Items:")
    general_faq = searcher.get_faq_by_category("general")
    for faq in general_faq:
        topic = faq['metadata'].get('topic', 'Unknown')
        print(f"   ❓ {topic}: {faq['id']}")
    
    # 4. Demonstrate collection-specific search
    print("\n🎯 Collection-Specific Search:")
    print("-" * 30)
    
    # Search only in code examples
    code_query = "sentiment analysis pipeline"
    code_results = searcher.search_specific_collection("code_examples", code_query, n_results=2)
    print(f"\n🔎 Searching code_examples for: '{code_query}'")
    if 'documents' in code_results:
        for i, doc in enumerate(code_results['documents'][0]):
            metadata = code_results['metadatas'][0][i]
            task = metadata.get('task', 'Unknown')
            difficulty = metadata.get('difficulty', 'Unknown')
            print(f"   💻 Result {i+1}: {task} ({difficulty})")
            print(f"      Preview: {doc[:80]}...")
    
    print("\n✅ Demo completed successfully!")
    print("💡 You can now use ChromaDB for intelligent document retrieval!")

if __name__ == "__main__":
    demo_search_functionality()
