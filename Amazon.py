import requests
from bs4 import BeautifulSoup as bs

headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"}
reviewslist = []

req = requests.get("https://amzn.eu/d/8OkdcSv")
soup = bs(req.content, 'html.parser')
content = soup.get_text()
reviewlink = soup.find("a", class_ = "a-link-emphasis a-text-bold")

if reviewlink:
        link = reviewlink['href'] if 'href' in reviewlink.attrs else None

def readreviews(link):
    reviewpage = "https://www.amazon.in/" + link
    print(reviewpage)
    req1 = requests.get(reviewpage)
    soup1 = bs(req1.content, 'html.parser')

    reviews = soup1.find_all("span", class_ = "a-size-base review-text review-text-content")
    reviewslist = []

    for review in reviews:
        review1 = review.get_text()
        review1 = review1.replace('\n', '')
        reviewslist.append(review1)
        print(review1)
    print(reviewslist)


    nextpageli = soup1.find("li", class_ = "a-last")
    if nextpageli:
        nextpage = nextpageli.find("a")
        link = nextpage['href'] if 'href' in nextpage.attrs else None
        print(link)
        readreviews(link)
    return reviewslist

reviewslis = readreviews(link)

print(len(reviewslis))

