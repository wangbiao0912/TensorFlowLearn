import requestsheaders = {    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36',    'cookie': 'juzi_user=642230; juzi_token=bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwczpcL1wvd3d3Lml0anV6aS5jb21cL2FwaVwvYXV0aG9yaXphdGlvbnMiLCJpYXQiOjE1ODU3MjczODIsImV4cCI6MTU4NTczMDk4MiwibmJmIjoxNTg1NzI3MzgyLCJqdGkiOiJuTlVpRkZTYmZKQm9sbER4Iiwic3ViIjo2NDIyMzAsInBydiI6IjIzYmQ1Yzg5NDlmNjAwYWRiMzllNzAxYzQwMDg3MmRiN2E1OTc2ZjciLCJ1dWlkIjoiWjlPekRpIn0.ulRlMMo16scpaE2cHdHes294aviMNUMDImmg2yn73dA',    'accept': 'application/json, text/plain, */*',    'accept-encoding': 'gzip, deflate, br',    'accept-language': 'zh-CN,zh;q=0.9',    'authorization': '"bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwczpcL1wvd3d3Lml0anV6aS5jb21cL2FwaVwvYXV0aG9yaXphdGlvbnMiLCJpYXQiOjE1ODU3MjczODIsImV4cCI6MTU4NTczMDk4MiwibmJmIjoxNTg1NzI3MzgyLCJqdGkiOiJuTlVpRkZTYmZKQm9sbER4Iiwic3ViIjo2NDIyMzAsInBydiI6IjIzYmQ1Yzg5NDlmNjAwYWRiMzllNzAxYzQwMDg3MmRiN2E1OTc2ZjciLCJ1dWlkIjoiWjlPekRpIn0.ulRlMMo16scpaE2cHdHes294aviMNUMDImmg2yn73dA"',    'cache-control': 'no-cache',    'content-length': '21',    'content-type': 'application/json;charset=UTF-8',    'Cookie': 'juzi_user=642230; juzi_token=bearer '              'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwczpcL1wvd3d3Lml0anV6aS5jb21cL2FwaVwvYXV0aG9yaXphdGlvbnMiLCJpYXQiOjE1ODU3MjczODIsImV4cCI6MTU4NTczMDk4MiwibmJmIjoxNTg1NzI3MzgyLCJqdGkiOiJuTlVpRkZTYmZKQm9sbER4Iiwic3ViIjo2NDIyMzAsInBydiI6IjIzYmQ1Yzg5NDlmNjAwYWRiMzllNzAxYzQwMDg3MmRiN2E1OTc2ZjciLCJ1dWlkIjoiWjlPekRpIn0.ulRlMMo16scpaE2cHdHes294aviMNUMDImmg2yn73dA',    'curlopt_followlocation': 'true',    'origin': 'https://www.itjuzi.com',    'pragma': 'no-cache',    'referer': 'https://www.itjuzi.com/',    'sec-fetch-dest': 'empty',    'sec-fetch-mode': 'cors',    'sec-fetch-site': 'same-origin'    }url = 'https://www.itjuzi.com/user/login?redirect=&flag=&radar_coupon='session = requests.Session()response = session.get('https://www.itjuzi.com/investevent', headers=headers)print(response.status_code)print(response.text)