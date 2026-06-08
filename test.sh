git add src
rm -rf /tmp/lanka_data/cache
python3 workflows/examples_build.py \
    && python3 workflows/readme_build.py \
    && python3 -m pytest -x -v -p no:warnings "$@" 


git add examples
git add README.md
git commit -m "Ran test.sh to update examples and README.md"