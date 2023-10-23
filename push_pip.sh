#!/bin/bash

echo Deleting existing dist folder
rm -rf dist
echo Enter desired version number
read version

cat pyproject.txt \
 | sed  "s#version = '0.0.1'#version='${version}'#g" \
 > pyproject.toml

python3 -m build

echo Upload new version $version to PyPi '(y,n)'?
read decision

if [ $decision = "Y" ] || [ $decision = 'y' ]; then
  twine upload dist/*
else
  echo Exiting program
fi