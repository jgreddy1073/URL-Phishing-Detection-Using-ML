import whois #fetches domain registration data (used for age/expiry features).
from urllib.parse import urlparse 
import httpx #makes the HTTP request to the actual URL (used for iframe/mouseover/forwarding features).
import pickle as pk #pickle loads your saved PCA model from disk.
import pandas as pd
import extractorFunctions as ef #ef is your own module — imported as an alias so you can call ef.getLength() etc.

#Function to extract features
def featureExtraction(url):

  features = [] #Empty basket, slowiy we put [],[len value],[ length value, depth value],...
  #Address bar based features (12)
  features.append(ef.getLength(url))
  features.append(ef.getDepth(url))
  features.append(ef.tinyURL(url))
  features.append(ef.prefixSuffix(url))
  features.append(ef.no_of_dots(url))
  features.append(ef.sensitive_word(url))
#Why: builds up one feature vector (a list of numbers) per URL, in a fixed order — this order has to exactly match what the model was trained on.

  domain_name = ''
  #Domain based features (4)
  dns = 0
  try:
    domain_name = whois.whois(urlparse(url).netloc)
  except:
    dns = 1

  features.append(1 if dns == 1 else ef.domainAge(domain_name))
  features.append(1 if dns == 1 else ef.domainEnd(domain_name))

  # HTML & Javascript based features (4)
  dom = []
  try:
    response = httpx.get(url)
  except:
    response = ""

  dom.append(ef.iframe(response))
  dom.append(ef.mouseOver(response))
  dom.append(ef.forwarding(response))

  features.append(ef.has_unicode(url)+ef.haveAtSign(url)+ef.havingIP(url))

  with open('model/pca_model.pkl', 'rb') as file:
    pca = pk.load(file)

  #converting the list to dataframe
  feature_names = ['URL_Length', 'URL_Depth', 'TinyURL', 'Prefix/Suffix', 'No_Of_Dots', 'Sensitive_Words',
                       'Domain_Age', 'Domain_End', 'Have_Symbol','domain_att']
  dom_pd = pd.DataFrame([dom], columns = ['iFrame','Web_Forwards','Mouse_Over'])
  features.append(pca.transform(dom_pd)[0][0])

  row = pd.DataFrame([features], columns= feature_names)

  return row
