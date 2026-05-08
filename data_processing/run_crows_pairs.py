import os
import json
import config
from data_processing.crows_pairs_generator import CrowSPairsGenerator


def main():
    print("=" * 60)
    print("Phase 1-4: CrowS-Pairs Format Contrastive Corpus Construction")
    print("=" * 60)

    crows_dir = config.ensure_dir(config.DATA_CONFIG["crows_pairs_dir"])

    bias_graph_path = os.path.join(config.DATA_CONFIG["chinese_bias_graph_dir"], "chinese_bias_graph.json")
    bias_graph = None
    if os.path.exists(bias_graph_path):
        with open(bias_graph_path, "r", encoding="utf-8") as f:
            bias_graph = json.load(f)
        print(f"Loaded bias graph from {bias_graph_path}")
    else:
        print("Using default ATTRIBUTE_PAIRS from config.py")

    generator = CrowSPairsGenerator(bias_graph=bias_graph, random_seed=config.HYPERPARAM_CONFIG["random_seed"])

    print("\n[Step 1] Generating CrowS-Pairs for each dimension...")
    for dimension in config.BIAS_DIMENSIONS:
        pairs = generator.generate_pairs(dimension, num_pairs=30)
        print(f"  {dimension}: {len(pairs)} pairs generated")

    print("\n[Step 2] Generating all pairs combined...")
    all_pairs = generator.generate_all_pairs(pairs_per_dimension=30)
    print(f"  Total: {len(all_pairs)} pairs")

    print("\n[Step 3] Saving to JSON...")
    output_path = generator.save_pairs(crows_dir, "crows_pairs_chinese.json")

    print("\n[Step 4] Statistics by dimension:")
    for dimension in config.BIAS_DIMENSIONS:
        dim_pairs = generator.get_pairs_by_dimension(dimension)
        print(f"  {dimension}: {len(dim_pairs)} pairs")

    print("\n[Sample Pairs]")
    for i, pair in enumerate(all_pairs[:3]):
        print(f"\n  Pair {i+1} ({pair['dimension']} - {pair['bias_type']}):")
        print(f"    Stereotype:     {pair['stereotype_sentence']}")
        print(f"    Anti-Stereotype: {pair['anti_stereotype_sentence']}")

    print("\n" + "=" * 60)
    print("Phase 1-4 completed!")
    print("=" * 60)
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
