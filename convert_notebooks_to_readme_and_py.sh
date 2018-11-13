rm Task1.py
rm Task2.py
rm -rf Task2_files/;
rm README.md

jupyter nbconvert --to python Task1.ipynb
jupyter nbconvert --to python Task2.ipynb
jupyter nbconvert --to markdown Task1.ipynb
jupyter nbconvert --to markdown Task2.ipynb

cat Task1.md > README.md
cat Task2.md >> README.md

rm Task1.md
rm Task2.md

