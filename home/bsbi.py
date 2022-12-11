import os
import pickle
import contextlib
import heapq
import time
import math

from .index import InvertedIndexReader, InvertedIndexWriter
from .util import IdMap, sorted_merge_posts_and_tfs
from .compression import StandardPostings, VBEPostings
from tqdm import tqdm
import nltk
import re

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.stem = nltk.stem.PorterStemmer()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.token = nltk.tokenize

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, os.path.dirname(__file__) +'\\index\\terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, os.path.dirname(__file__) +'\\index\\docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, os.path.dirname(__file__) +'\\index\\terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, os.path.dirname(__file__) +'\\index\\docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)
        with InvertedIndexReader(
                directory=self.output_dir,
                index_name=self.index_name,
                postings_encoding=self.postings_encoding
            ) as inverted_map:
            self.doc_length = inverted_map.doc_length

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        # TODO v        
        
        term_doc_pairs = set({})
        file_list = os.listdir(os.path.join(self.data_dir,block_dir_relative))

        for file in file_list:
            file_dir = os.path.join(block_dir_relative,file)
            open_as_text = open(os.path.join(self.data_dir, file_dir)).read()
            open_as_text = open_as_text.lower()
            open_as_text = re.sub("\s+", " ", open_as_text) # Menghilangkan spasi berlebih
            open_as_text = re.sub("[^\w\s]", " ", open_as_text) # Menghilangkan tanda baca'
            words = self.token.word_tokenize(open_as_text)
            words = [w for w in words if w not in self.stopwords]
            words = [self.stem.stem(w) for w in words]
            docID = self.doc_id_map.__getitem__(file_dir)

            for word in words:
                termID = self.term_id_map.__getitem__(word)
                term_doc_pairs.add((termID,docID))
        
        return list(term_doc_pairs)

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        # TODO v

        term_dict = dict()
        for term_id, doc_id in td_pairs:
            term_dict.setdefault(term_id, dict())
            term_dict[term_id].setdefault(doc_id, 0)
            term_dict[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            doc_fq_pairs = sorted(term_dict[term_id].items())
            unzipped = list(zip(*doc_fq_pairs))
            index.append(term_id, list(unzipped[0]), list(unzipped[1]))

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_bm25(self, query, k1=2.745, b=0.75, k = 100):
        self.load()
        
        queries = query.lower()
        queries = re.sub("\s+", " ", queries) # Menghilangkan spasi berlebih
        queries = re.sub("[^\w\s]", " ", queries) # Menghilangkan tanda baca'
        words = self.token.word_tokenize(queries)
        words = [w for w in words if w not in self.stopwords]
        queries = [self.stem.stem(w) for w in words]
        res = dict()
        with InvertedIndexReader(
                directory=self.output_dir,
                index_name=self.index_name,
                postings_encoding=self.postings_encoding
            ) as inverted_map:
            for term in queries:
                if term in self.term_id_map:
                    pl,tl=inverted_map.get_postings_list(self.term_id_map[term])
                    for i in range(len(pl)):
                        m = len(self.doc_id_map)
                        tf = tl[i]
                        df = inverted_map.postings_dict[pl[i]][1]
                        dl = self.doc_length[pl[i]]
                        avg = 0
                        count = 0
                        for ele in self.doc_length:
                            avg += ele
                            count += 1
                        avg_count = avg/count
                        wtd=((k1 + 1) * tf) / (k1 * ((1 - b) + b * dl / avg_count) + tf)  
                        if self.doc_id_map[pl[i]] not in res.values():
                            res[self.doc_id_map[pl[i]]] = (wtd*math.log(m/df))
                        else:
                            res[self.doc_id_map[pl[i]]] += (wtd*math.log(m/df))
        res_val = res.values()
        res_key = res.keys()
        result = list(zip(res_val,res_key))
        result = sorted(result, key=lambda i : i[0],reverse=True)  
        return result[:k]                                                                         
                

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = VBEPostings, \
                              output_dir = 'index')
    BSBI_instance.index() # memulai indexing!
