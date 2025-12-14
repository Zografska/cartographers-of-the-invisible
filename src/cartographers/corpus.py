import requests
import time
import re
from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple
import json

class SemanticCorpusExtractor:
    """
    Extract semantic corpus from ConceptNet with Gradio fallback
    """
    
    def __init__(self, use_gradio_fallback=True):
        self.api_base = "https://api.conceptnet.io"
        self.use_gradio_fallback = use_gradio_fallback
        self.gradio_url = "https://cstr-conceptnet-normalized.hf.space"
        self.fallback_client = None
        self.cache = {}
        
    def _init_fallback(self):
        """Initialize Gradio client for fallback"""
        if self.fallback_client is None and self.use_gradio_fallback:
            try:
                from gradio_client import Client
                self.fallback_client = Client(self.gradio_url)
                print("✓ Gradio fallback initialized")
            except Exception as e:
                print(f"⚠ Could not initialize Gradio fallback: {e}")
                self.use_gradio_fallback = False
    
    def get_related_concepts(self, word: str, language: str = "en", 
                            relations: List[str] = None, 
                            limit: int = 100) -> Dict:
        """
        Get related concepts using primary API or Gradio fallback
        
        Args:
            word: Target word to extract semantic domain for
            language: Language code (default: 'en')
            relations: List of relation types to query
            limit: Maximum results per relation
        """
        if relations is None:
            relations = ['RelatedTo', 'IsA', 'PartOf', 'UsedFor', 
                        'HasProperty', 'AtLocation', 'CapableOf']
        
        # Try cache first
        cache_key = f"{word}_{language}_{'_'.join(sorted(relations))}"
        if cache_key in self.cache:
            print(f"📦 Using cached result for '{word}'")
            return self.cache[cache_key]
        
        # Try direct API first
        result = self._query_conceptnet_api(word, language, relations, limit)
        
        # If API fails, try Gradio fallback
        if result["source"] == "error" and self.use_gradio_fallback:
            print(f"⚠ API failed, trying Gradio fallback for '{word}'...")
            result = self._query_gradio_fallback(word, language, relations)
        
        # Cache the result if successful
        if result["source"] != "error":
            self.cache[cache_key] = result
        
        return result
    
    def _query_conceptnet_api(self, word: str, language: str, 
                              relations: List[str], limit: int) -> Dict:
        """Query ConceptNet API directly"""
        try:
            all_edges = []
            related_concepts = set()
            relation_results = {}
            
            for relation in relations:
                url = f"{self.api_base}/query"
                params = {
                    'start': f'/c/{language}/{word.replace(" ", "_")}',
                    'rel': f'/r/{relation}',
                    'limit': limit
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    edges = data.get('edges', [])
                    all_edges.extend(edges)
                    relation_results[relation] = len(edges)
                    
                    # Extract related concepts
                    for edge in edges:
                        end_label = edge.get('end', {}).get('label', '')
                        if end_label:
                            related_concepts.add(end_label)
                    
                    time.sleep(0.1)  # Rate limiting
                    
                elif response.status_code == 502:
                    raise Exception("API returned 502 Bad Gateway")
                else:
                    print(f"⚠ Status {response.status_code} for relation {relation}")
            
            return {
                "source": "conceptnet_api",
                "word": word,
                "language": language,
                "relations_queried": relations,
                "relation_counts": relation_results,
                "edges": all_edges,
                "related_concepts": list(related_concepts),
                "count": len(related_concepts)
            }
            
        except Exception as e:
            print(f"❌ API error: {e}")
            return {
                "source": "error",
                "error": str(e),
                "word": word
            }
    
    def _query_gradio_fallback(self, word: str, language: str, 
                               relations: List[str]) -> Dict:
        """Use Gradio interface as fallback"""
        try:
            self._init_fallback()
            
            if self.fallback_client is None:
                raise Exception("Gradio client not available")
            
            print(f"🔄 Querying Gradio for '{word}'...")
            
            # Call the Gradio interface
            result = self.fallback_client.predict(
                word=word,
                lang=language,
                selected_relations=relations,
                api_name="/get_semantic_profile"
            )
            
            # Parse the markdown result
            concepts, weights = self._parse_markdown_result(result)
            
            print(f"✓ Gradio returned {len(concepts)} concepts")
            
            return {
                "source": "conceptnet_normalized",
                "word": word,
                "language": language,
                "relations_queried": relations,
                "raw_data": result,
                "related_concepts": concepts,
                "concept_weights": weights,
                "count": len(concepts)
            }
            
        except Exception as e:
            print(f"❌ Gradio fallback failed: {e}")
            return {
                "source": "error",
                "error": f"Both API and Gradio failed: {str(e)}",
                "word": word
            }
    
    def _parse_markdown_result(self, markdown_text: str) -> Tuple[List[str], Dict[str, float]]:
        """
        Parse markdown formatted result from Gradio
        
        Expected format:
        - **coffee** RelatedTo → *latte* `[1.333]`
        - *sugar* RelatedTo → **coffee** `[2.655]`
        
        Returns:
            Tuple of (concepts_list, weights_dict)
        """
        concepts = []
        weights = {}
        
        # Pattern to match lines like:
        # - **coffee** RelatedTo → *latte* `[1.333]`
        # - *sugar* RelatedTo → **coffee** `[2.655]`
        pattern = r'-\s+(?:\*\*[\w\s]+\*\*|\*[\w\s]+\*)\s+\w+\s+→\s+\*(?:\*)?([^*`]+?)(?:\*)?(?:\*)?`\s*\[([0-9.]+)\]`'
        
        # More flexible pattern to catch concept names
        # Matches text between asterisks: *concept* or **concept**
        concept_pattern = r'\*+([^*]+?)\*+'
        
        lines = markdown_text.split('\n')
        
        for line in lines:
            if '→' in line and ('*' in line):
                # Extract all concepts (text between asterisks)
                found_concepts = re.findall(concept_pattern, line)
                
                # Extract weight if present
                weight_match = re.search(r'\[([0-9.]+)\]', line)
                weight = float(weight_match.group(1)) if weight_match else 1.0
                
                for concept in found_concepts:
                    concept = concept.strip()
                    # Skip the source word (usually in bold)
                    if concept and not concept.startswith('**'):
                        concepts.append(concept)
                        # Store weight for this concept
                        if concept not in weights or weights[concept] < weight:
                            weights[concept] = weight
        
        # Remove duplicates while preserving order
        seen = set()
        unique_concepts = []
        for c in concepts:
            if c not in seen:
                seen.add(c)
                unique_concepts.append(c)
        
        return unique_concepts, weights
    
    def build_semantic_corpus(self, seed_words: List[str], 
                             max_depth: int = 2,
                             min_frequency: int = 2,
                             max_concepts_per_word: int = 50,
                             min_weight: float = 0.0) -> Dict:
        """
        Build a semantic corpus from seed words by traversing relations
        
        Args:
            seed_words: Starting words for corpus building
            max_depth: How many hops to traverse
            min_frequency: Minimum occurrences to include in corpus
            max_concepts_per_word: Limit concepts per word to prevent explosion
            min_weight: Minimum weight threshold for concepts
        """
        corpus = defaultdict(int)
        weights = defaultdict(float)
        visited = set()
        current_level = set(seed_words)
        metadata = {
            "errors": [],
            "skipped": [],
            "depth_stats": []
        }
        
        for depth in range(max_depth):
            print(f"\n📍 Depth {depth + 1}/{max_depth}: Processing {len(current_level)} words...")
            next_level = set()
            processed = 0
            failed = 0
            
            for word in current_level:
                if word in visited:
                    continue
                
                visited.add(word)
                corpus[word] += 1
                
                # Get related concepts
                result = self.get_related_concepts(word)
                
                if result["source"] != "error":
                    # Extract related concepts
                    related = result.get("related_concepts", [])
                    concept_weights = result.get("concept_weights", {})
                    
                    # Filter by weight if specified
                    if min_weight > 0.0 and concept_weights:
                        related = [c for c in related if concept_weights.get(c, 0) >= min_weight]
                    
                    # Limit concepts per word
                    if len(related) > max_concepts_per_word:
                        # Sort by weight if available
                        if concept_weights:
                            related = sorted(related, key=lambda c: concept_weights.get(c, 0), reverse=True)
                        related = related[:max_concepts_per_word]
                    
                    for concept in related:
                        corpus[concept] += 1
                        # Update weight (keep max weight seen)
                        if concept in concept_weights:
                            weights[concept] = max(weights[concept], concept_weights[concept])
                        
                        if concept not in visited and len(next_level) < 1000:  # Hard limit
                            next_level.add(concept)
                    
                    processed += 1
                    print(f"  ✓ {word}: found {len(related)} related concepts")
                else:
                    failed += 1
                    metadata["errors"].append({
                        "word": word,
                        "depth": depth + 1,
                        "error": result.get("error", "Unknown")
                    })
                    print(f"  ✗ {word}: failed")
                
                time.sleep(0.3)  # Be nice to APIs
            
            metadata["depth_stats"].append({
                "depth": depth + 1,
                "processed": processed,
                "failed": failed,
                "next_level_size": len(next_level)
            })
            
            current_level = next_level
            
            if not current_level:
                print(f"⚠ No more concepts to explore at depth {depth + 1}")
                break
        
        # Filter by minimum frequency
        filtered_corpus = {
            word: {
                "frequency": freq,
                "weight": weights.get(word, 0.0)
            }
            for word, freq in corpus.items() 
            if freq >= min_frequency
        }
        
        return {
            "corpus": filtered_corpus,
            "total_concepts": len(filtered_corpus),
            "total_visited": len(visited),
            "raw_corpus_size": len(corpus),
            "depth_reached": depth + 1,
            "seed_words": seed_words,
            "metadata": metadata
        }
    
    def export_corpus(self, corpus_data: Dict, filename: str):
        """Export corpus to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Corpus exported to {filename}")
    
    def export_simple_list(self, corpus_data: Dict, filename: str):
        """Export just the list of concepts to a simple text file"""
        concepts = sorted(corpus_data['corpus'].keys())
        with open(filename, 'w', encoding='utf-8') as f:
            for concept in concepts:
                f.write(f"{concept}\n")
        print(f"✓ Concept list exported to {filename}")
    
    def get_corpus_stats(self, corpus_data: Dict):
        """Print detailed statistics about the corpus"""
        print("\n" + "="*60)
        print("📊 CORPUS STATISTICS")
        print("="*60)
        print(f"Seed words: {', '.join(corpus_data['seed_words'])}")
        print(f"Total concepts (filtered): {corpus_data['total_concepts']}")
        print(f"Total visited: {corpus_data['total_visited']}")
        print(f"Raw corpus size: {corpus_data['raw_corpus_size']}")
        print(f"Depth reached: {corpus_data['depth_reached']}")
        
        # Depth statistics
        print("\n📈 Depth Statistics:")
        for stat in corpus_data['metadata']['depth_stats']:
            print(f"  Depth {stat['depth']}: {stat['processed']} processed, "
                  f"{stat['failed']} failed, {stat['next_level_size']} queued")
        
        # Frequency distribution
        corpus = corpus_data['corpus']
        freq_dist = defaultdict(int)
        for word, data in corpus.items():
            freq = data['frequency'] if isinstance(data, dict) else data
            freq_dist[freq] += 1
        
        print("\n📊 Frequency Distribution:")
        for freq in sorted(freq_dist.keys(), reverse=True)[:10]:
            print(f"  Frequency {freq}: {freq_dist[freq]} words")
        
        # Top concepts by frequency
        sorted_by_freq = sorted(
            corpus.items(), 
            key=lambda x: x[1]['frequency'] if isinstance(x[1], dict) else x[1], 
            reverse=True
        )
        print("\n🏆 Top 20 Concepts by Frequency:")
        for i, (word, data) in enumerate(sorted_by_freq[:20], 1):
            if isinstance(data, dict):
                print(f"  {i:2d}. {word}: freq={data['frequency']}, weight={data['weight']:.3f}")
            else:
                print(f"  {i:2d}. {word}: {data}")
        
        # Top concepts by weight
        if any(isinstance(data, dict) and 'weight' in data for data in corpus.values()):
            sorted_by_weight = sorted(
                corpus.items(),
                key=lambda x: x[1].get('weight', 0) if isinstance(x[1], dict) else 0,
                reverse=True
            )
            print("\n⭐ Top 20 Concepts by Weight:")
            for i, (word, data) in enumerate(sorted_by_weight[:20], 1):
                if isinstance(data, dict):
                    print(f"  {i:2d}. {word}: weight={data['weight']:.3f}, freq={data['frequency']}")
        
        # Errors
        errors = corpus_data['metadata']['errors']
        if errors:
            print(f"\n⚠ Errors encountered: {len(errors)}")
            for error in errors[:5]:
                print(f"  - {error['word']} (depth {error['depth']}): {error['error']}")
    
    def get_all_concepts(self, corpus_data: Dict) -> List[str]:
        """Extract just the list of concept names"""
        return sorted(corpus_data['corpus'].keys())


# Example usage
if __name__ == "__main__":
    print("🚀 ConceptNet Semantic Corpus Extractor")
    print("="*60)
    
    # Initialize extractor with Gradio fallback
    extractor = SemanticCorpusExtractor(use_gradio_fallback=True)
    
    # Example 1: Extract semantic domain for a single word
    print("\n=== Single Word Extraction ===")
    result = extractor.get_related_concepts(
        word="coffee",
        relations=['RelatedTo', 'IsA', 'UsedFor']
    )
    
    print(f"\nSource: {result.get('source')}")
    print(f"Found {result.get('count', 0)} related concepts")
    
    if result.get('count', 0) > 0:
        concepts = result['related_concepts']
        print(f"\n📋 All {len(concepts)} extracted concepts:")
        for i, concept in enumerate(concepts, 1):
            weight = result.get('concept_weights', {}).get(concept, 0)
            print(f"  {i:2d}. {concept} (weight: {weight:.3f})")
    
    # Example 2: Build semantic corpus from seed words
    print("\n\n=== Building Semantic Corpus ===")
    seed_words = ["coffee", "tea"]
    
    corpus_data = extractor.build_semantic_corpus(
        seed_words=seed_words,
        max_depth=2,
        min_frequency=2,
        max_concepts_per_word=30
    )
    
    # Show statistics
    extractor.get_corpus_stats(corpus_data)
    
    # Get all concepts as array
    all_concepts = extractor.get_all_concepts(corpus_data)
    print(f"\n📋 COMPLETE CONCEPT ARRAY ({len(all_concepts)} items):")
    print(all_concepts[:50])  # Show first 50
    
    # Export corpus
    print("\n")
    extractor.export_corpus(corpus_data, "semantic_corpus.json")
    extractor.export_simple_list(corpus_data, "concepts_list.txt")
    
    print("\n✅ Done!")