from flask import Flask, render_template, request,flash
import pandas as pd 
from functionMT import Get_Ans,Select_RankingFunction,EngAnsToThai

app = Flask(__name__)
app.secret_key = "super secret key"

#setup context dataset
meta_df = pd.read_csv('./Cord_Abs_Df21.csv')
Cord_Corpus = meta_df['abstract'].values.tolist()
Cord_Corpus_str = [str(element) for element in Cord_Corpus]
tokenized_corpus = [doc.split(" ") for doc in Cord_Corpus_str]

def test():
    print("test")


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        #get Question text from Html page
        Query = request.form['Question']
        tokenized_query = Query.split(" ")
    
        #select bm25 fucntion base on user choice
        bm25fn = request.form['bm25Fn']
        #bm25 = BM25Plus(tokenized_corpus)
        bm25 = Select_RankingFunction(bm25fn,tokenized_corpus)
        #get number of topN document
        topN = request.form['DocN']
        topN = int(topN)
        doc_scores = bm25.get_scores(tokenized_query)
        Context = bm25.get_top_n(tokenized_query, Cord_Corpus, n=topN)

        ans_df = pd.DataFrame()
        ans_df = Get_Ans(Query,Context,topN)[0]
        lang = Get_Ans(Query,Context,topN)[1]
        if lang == 'else':
            flash('This WebApplication accept only Thai and English text')
            return render_template('index copy.html')
        meta_df['doc_scores'] = doc_scores
        cABS = meta_df.sort_values(by=['doc_scores'],ascending=False)
        cABS=cABS.head(topN)
        ans_df['Thai answer'] = EngAnsToThai(ans_df['answer'])
        ans_df['title'] = cABS['title'].values
        ans_df['Abstract'] = cABS['abstract'].values
        ans_df['doc_scores'] = cABS['doc_scores'].values
        ans_df['Url'] = cABS['url'].values
        ans_df = ans_df.sort_values(by=['score'],ascending=False)
        ans_df = ans_df.reset_index(drop=True)

        def create_link(url):
            return f'<a href="{url}" target="_blank">{url}</a>'
        
        # ans_df['Url'] = ans_df['Url'].apply(create_link)

        ans_df = ans_df.drop(columns=['score', 'start','end','doc_scores'])
        # return render_template('index.html', data=ans_df.to_html(escape=False,classes='table table-striped'), titles=" ",query = Query)
        data = ans_df.to_dict('records')
        return render_template('index.html', data=data,query = Query)
    else:
        return render_template('index.html')
    

if __name__ == '__main__':
    app.run(debug=True)