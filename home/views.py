from django.shortcuts import render
import sys
import os
from .myModel import eval_whole
from datetime import datetime

# Create your views here.

def index(request):
    query = request.GET.get('search_box')

    start_time = datetime.now()
    totalTime=None
    
    if query == None or query == "":
        context = {
            'query': query,
            'flag': -1
        }
        return render(request, 'home/index.html', context)
    else:
        result = {}
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        result_raw = eval_whole(100, query)
        flag = 0

        if result_raw != None:       
            for (doc_id, _) in result_raw:
                col = int(doc_id)//100 + 1
                text_file = os.path.dirname(__file__) + "/collection/" + str(col) + "/" + str(doc_id) +".txt"
                text = open(text_file).read()
                text = text.lower()
                result[doc_id] = text
            flag = 1

        end_time = datetime.now()
        totalTime = end_time-start_time

        context = {
            'result': result,
            'query': query,
            'flag': flag,
            'totalTime' : totalTime,
            'lenResult' : len(result)
        }
        
        return render(request, 'home/index.html', context)

def detail(request, doc_id):
    n = int(doc_id)//100 + 1
    dir_text = os.path.dirname(__file__) + "/collection/" + str(n) + "/" + str(doc_id) +".txt"
    text = open(dir_text).read()

    context = {
        'doc_id': doc_id,
        'text': text,
    }
   
    return render(request, 'home/detail.html', context)