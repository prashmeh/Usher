import json
import csv

# with open('data.json') as json_file:
# Bring your own data.json
with open('data.json') as json_file:
    data = json.load(json_file)
    row = ['tags', 'post']

    # with open('zendeskData.csv', 'a') as csvFile:
    #     writer = csv.writer(csvFile)
    #     writer.writerow(row)
    #     count = 0
    #     for p in data:
    #         tag = p['fields'][0]['value']
    #         if tag != 'extensions':
    #             count += 1
    #             if count == 10:
    #                     count = 0
    #                     row = ['non-extension', p['description'].encode('utf-8')]
    #                     writer.writerow(row)
    #         else:
    #             row = [p['fields'][0]['value'], p['description'].encode('utf-8')]
    #             writer.writerow(row)
    with open('zendeskData.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
        for p in data:
            row = [p['fields'][0]['value'], p['description'].encode('utf-8')]
            writer.writerow(row)

    csvFile.close()
