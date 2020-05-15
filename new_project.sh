#!/bin/bash
folders=("data"
	 "doc"
	 "src")

mkdir -p $1

# $@ <=> all arguments passed to script $1, $2, ... etc

for folder in "${folders[@]}"; do
	mkdir -p $1/$folder
done
