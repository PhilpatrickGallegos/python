import requests
from bs4 import BeautifulSoup

#Get the website link
r = requests.get('https://coinmarketcap.com/all/views/all/')

#Get text from website in HTML format
soup = BeautifulSoup(r.text, 'html.parser')

#Initalize list 
data = []

#Grab table with all currency information
table = soup.find('table', id='currencies-all')

#Grab the <tr> HTML row
#We can grab (name,symbol,marketcap,price,total supply,
#             volume,change 1h, change 24h, change 7d)
for row in table.find_all('tr'):
#Try and except incase of error.
    try:
#Find the Crypto information
        symbol = row.find('td', class_='text-left col-symbol').text
        price = row.find('a', class_='price').text
        time_1h = row.find('td', {'data-timespan': '1h'}).text
        time_24h = row.find('td', {'data-timespan': '24h'}).text
        time_7d = row.find('td', {'data-timespan': '7d'}).text
#If an error occured keep going
    except AttributeError:
        continue
#Store everything in our list
    data.append((symbol, price, time_1h, time_24h, time_7d))

#Print information
for item in data:
    print(item)
