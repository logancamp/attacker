.PHONY: all sample phase0 phase1 phase2 phase3 phase4 phase5 phase6 clean

clean:
	rm -rf output pipeline_ready.csv
	rm -f data/aol_sample.csv data/aol_queries_only.csv data/pipeline_ready_base.csv

# Samples users from the raw AOL dataset -> aol_sample.csv
# Adds Label/Role columns to aol_sample.csv, writes pipeline_ready_base.csv
sample:
	python sample_aol.py \
		--input_path data/user-ct-test-collection-02.txt \
		--output_csv data/aol_sample.csv \
		--output_queries_only_csv data/aol_queries_only.csv \
		--target_users 1000 \
		--max_queries_per_user 100 \
		--session_gap_minutes 30

# Reads pipeline_ready_base.csv (real only), writes pipeline_ready.csv (real + fakes)
gen_fakes:
	python phase1.py \
		--input data/aol_sample.csv \
		--output data/pipeline_ready_base.csv

	python gen_fakes.py \
 		--input data/pipeline_ready_base.csv \
 		--output data/pipeline_ready.csv

# TODO: add obfuscation methods here
# Create and inject fakes, writes with real+fakes
# Apply DP-Comet obfuscation, writes with real+fakes

# Cleans the RQI output, writes aol_rqi_ratio_1_1_english_spanish_french_1-3.csv from fakes additon
phase1:
	python temp_clean.py \
		--input data/aol_rqi_ratio_1_1_english_spanish_french_1-3.csv \
		--output data/pipeline_ready.csv

# Extracts 16 features per query
phase2:
	python phase2_feature_extraction.py \
		--input data/pipeline_ready.csv \
		--output_dir output

# Builds pairwise features for training and attack
phase3:
	python phase3_pairwise_features.py \
		--features output/query_features.pkl \
		--mode train \
		--output_dir output
	python phase3_pairwise_features.py \
		--features output/query_features.pkl \
		--mode target \
		--output_dir output

# Trains 60 GBRT models on training pairs
phase4:
	python phase4_gbrt_training.py \
		--pairs output/train_pairs.pkl \
		--output_dir output

# Runs linkage attack, k-means k=2, computes FP/FN
phase5:
	python phase5_linkage_attack.py \
		--target_pairs output/target_pairs.pkl \
		--models_dir output/models \
		--output_dir output

# Builds topic profiles and saves human-readable summary
phase6:
	python phase6_results.py \
		--cluster_results output/cluster_results.csv \
		--attack_metrics output/attack_metrics.pkl \
		--output_dir output

all_easy_fakes: sample gen_fakes phase2 phase3 phase4 phase5 phase6
all_injected_fakes: phase1 phase2 phase3 phase4 phase5 phase6 #add sample and obfuscation steps when ready