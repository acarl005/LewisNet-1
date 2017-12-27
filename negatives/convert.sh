# take multi-page PDFs and convert to one-page PNGs
convert "$1" -background white -alpha remove -resize 1200x1600 -quality 100 -density 150 "img-pg/$(basename "$1" .pdf).png"

