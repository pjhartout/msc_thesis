format:
	isort .
	black .

download_human_proteome_alphafold:
	wget https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v2.tar -P data/
	tar -cvzf UP000005640_9606_HUMAN_v2.tar UP000005640_9606_HUMAN_v2/
	cd UP000005640_9606_HUMAN_v2/
	gzip -d *.gz
