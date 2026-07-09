
git add src


rm -rf /tmp/lanka_data/cache
rm -rf _output
python3 workflows/examples_build.py \
    && python3 workflows/readme_build.py \
    && python3 -m pytest -x -v -p no:warnings "$@" 


