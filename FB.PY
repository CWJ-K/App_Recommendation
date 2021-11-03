"""
Created on Thursday  Feb 21, 2019
@title : Media and Web Analytics
@Group : 2 - (Chen Wan Ju, Sweetie Anang, Gandes Aisyaharum, Nessya Callista, Mirza Miftanula)
@author: Chen Wan Ju
@Obj   : Scrape FB comments 
"""



from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import WebDriverException
import pandas as pd
import time


# function for find "more comments"
def find(browser):
    element = browser.find_elements_by_partial_link_text('more comments')
    if element:
        return element
    else:
        return False

def find2(browser):
    element2 = browser.find_elements_by_partial_link_text('See more')
    if element2:
        return element2
    else:
        return False


browser = webdriver.Chrome(r'C:/Users/R/Desktop/chromedriver')
browser.get('https://www.facebook.com/')

# Login Facebook ==> avoid to jumping out many windows of FB
username= "group2Mediaweb@gmail.com"
password = "123hello456"

User = browser.find_element_by_id('email')
User.send_keys(username)

Pass = browser.find_element_by_id('pass')
Pass.send_keys(password)

browser.find_element_by_id('loginbutton').click()

#Enter the FB page of Curry Pc World
browser.get('https://www.facebook.com/curryspcworld/')



## Remove: without removing, python can not click the buttons of inner pages
# Remove the windows of Chat 
Chat = "document.getElementById('BuddylistPagelet').remove();"
browser.execute_script(Chat)
# Remove the blue bar with account and the icon of FB
bluebar = "document.getElementById('pagelet_bluebar').remove();"
browser.execute_script(bluebar)

bluebarRoot="document.getElementById('bluebarRoot').remove();"
browser.execute_script(bluebarRoot)

body = browser.find_element_by_tag_name('body')

# if error: not clickable appears: use "inspect" to resist a block, 
# since the block of Fb: like, follow, share, will affect the process 
for _ in range(2000):
    try:
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(2)
        WebDriverWait(browser, 8).until_not(EC.presence_of_element_located(
            (By.CSS_SELECTOR, '.mls.img._55ym._55yn._55yo')))
        try:
            if  WebDriverWait(browser, 8).until(find) or WebDriverWait(browser, 8).until(find2):
                browser.find_element_by_partial_link_text('more comments').click()
                time.sleep(1)
                browser.find_element_by_partial_link_text('See more').click()
                time.sleep(1)
        except NoSuchElementException:
            for _ in range(0.5):
                body.send_keys(Keys.PAGE_DOWN)
                time.sleep(1)
        except StaleElementReferenceException:
            for _ in range(0.5):
                body.send_keys(Keys.PAGE_DOWN)
                time.sleep(1)
                #except WebDriverException:
                    #for _ in range(0.5):
                        #body.send_keys(Keys.PAGE_DOWN)
                        #time.sleep(0.5)
    except TimeoutException:
        print("Wait")    
  

html_source = browser.page_source
soup = BeautifulSoup(html_source, 'html.parser')

review_box = soup.find_all('div', attrs={'class': 'UFICommentActorAndBodySpacing'})

comment = []
for d in review_box:
    comment_text = d.find('span',attrs={'class':'UFICommentBody'}) 
    comment.append(comment_text.text)

ID = []
for user in browser.find_elements_by_class_name('UFICommentActorName'):
    ID.append(user.text)

df = pd.DataFrame({'ID':ID,'Comment':comment},index=range(1,len(ID)+1))
df = df[df.ID!="Currys PC World"]
df.to_excel("C:/Users/R/Desktop/FB.xlsx",sheet_name='FB')

