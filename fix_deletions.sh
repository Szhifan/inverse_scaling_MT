#!/bin/bash

# specify the commit hash
commit_hash=29fb3ecdf051ef6cd00002e8383b30bc4e4594cc

# get the list of deleted files
files=$(git diff --diff-filter=DM --name-only $commit_hash)

# loop through the files and restore them
for file in $files
do
  git checkout -p $commit_hash -- $file
done

