#!/bin/bash

# specify the commit hash
commit_hash=c9d2403

# get the list of deleted files
files=$(git diff --diff-filter=DM --name-only $commit_hash)

# loop through the files and restore them
for file in $files
do
  git checkout -p $commit_hash -- $file
done

