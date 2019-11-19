import requests

from research.gitdata import GitData

# %%
'''
Github search
'''
result = []
for i in range(1, 4):
    url = "https://api.github.com/search/repositories?q=federated+learning&per_page=100&page=" + str(i)
    json_data = requests.get(url)
    data = json_data.json()

    for i in data["items"]:
        result.append(i)

    # result.append(data["items"])

print(len(result))
# %%
git_result = []

for i in result:
    full_name = i["full_name"].split('/')

    repo = full_name[0]
    name = full_name[1]
    html_url = i["html_url"]
    desc = i["description"]
    language = i["language"]
    forks_count = i['forks_count']
    licenses = i['license']

    if licenses is None:
        licenses = {"name": "None"}

    watchers = i['watchers']
    score = i['score']

    print("{} / {} / {} / {} / {} / {} / {} / {} / {}".format(
        repo, name, html_url, desc, language, forks_count, licenses["name"], watchers, score))

    gitdata = GitData(repo=repo, name=name, html_url=html_url, desc=desc,
                      language=language, forks_count=forks_count, license=licenses['name'],
                      watchers=watchers, score=score)
    git_result.append(gitdata)

print(git_result)

# %%
import pandas as pd

df = pd.DataFrame([vars(s) for s in git_result])
df.to_csv("research/research_git.csv")

