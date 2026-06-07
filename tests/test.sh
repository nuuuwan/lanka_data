git add src
git commit -m "Various src changes"

rm -rf /tmp/lanka_data/cache
python3 workflows/examples_build.py \
    && python3 workflows/readme_build.py \
    && python3 -m pytest -x -v -p no:warnings "$@" 


git add .
git commit -m "Updated examples and readme"

git push origin main