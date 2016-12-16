import pickle
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import ngrams
import re
import editdistance
import json
import csv
from multiprocessing import Process, Queue
from datetime import datetime
import heapq
from lxml import etree


class PeriodIndex:
    def __init__(self, mapping_csv, member_json, validation_set='validationnames.csv',
                 name_mapping='', delimiter=','):
        self.index = {}

        with open(member_json) as fle:
            periods = json.load(fle)
            for period in periods:
                years = range(period['start'], period['end'])
                self.index[years] = {'members': period['members'],
                                     'files': {},
                                     'validation': [],
                                     'not_names' : ['DESREPRÉSENTANTS.',
                                                    'RESREPRÉSENTANTS.',
                                                    'CHAMBRE DES REPRÉSENTANTS.',
                                                    "ià'lG CHAMBRE DES REPRÉSENTANTS.",
                                                    "REPRESENTANTS",
                                                    u"Séance",
                                                    "Art.",
                                                    "SOMMAIRE",
                                                    u"SENTÀNTS",
                                                    "CHAMBRE",
                                                    u"Excédent",
                                                    u"Dépenses",
                                                    u"REPRÉSKNTANTS",
                                                    u"Section",
                                                    u"Notes du gouvernement.",
                                                    u"$",
                                                    u".....",
                                                    u"§",
                                                    u"«","CHAPITRE",
                                                    'DISREPRESENTANTS.'],
                                     'name_strip': ["—AnalysedespiècesadresséesàlaChambre.—Motiond'ordiede",
                                                    "dospiècesadresséesàlaeliambre.—Voleduprojetdeloitendantàprorogerlaloidujuilletsurlesconcessionsdepéages",
                                                    ]
                                     }

        with open(validation_set) as fle:
            validation = csv.reader(fle, delimiter=delimiter, quotechar='"')
            for year, speaker, name in validation:
                self[int(year)]['validation'].append((speaker, name))

        with open(mapping_csv) as fle:
            lines = [l for l in csv.reader(fle, delimiter=delimiter)]
            for line in lines:
                fname, date, page_num, url = line
                fname = fname.replace('PDF', 'txt')
                if date != 'indices':
                    if len(date) < 10:
                        date = '01/01/' + date
                    date = datetime.strptime(date, '%d/%m/%Y')
                    period = self[date.year]
                    if period:
                        period['files'][fname] = {'date': date.strftime('%d/%m/%Y'),
                                                  'pagenum': page_num,
                                                  'url': url,
                                                  }
                    else:
                        print(fname, date, 'file not found')

        # Add the proper abbreviations.
        if name_mapping:
            with open(name_mapping) as fle:
                lines = [l for l in csv.reader(fle, delimiter=delimiter)]
                for full_name, full_name2, abbr in lines:
                    if abbr and abbr != '...':
                        for year in self.index.values():
                            for member in year['members']:
                                if member['name'] == full_name or member['name'] == full_name2:
                                    member['abbr'] = abbr

    def __getitem__(self, year):
        for period, value in self.index.items():
            if year in period:
                return value
        print(year)

    def get_speakers(self, year):
        return [s['name'] for s in self[year].get('members', [])]


class Worker(Process):
    def __init__(self, in_q, name_distance=True, iter_train=0, output_path='./'):
        super(Worker, self).__init__()
        self.in_q = in_q
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 6),
                                          lowercase=False)
        self.ngramvectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1))
        self.classifier = svm.SVC(kernel='linear', probability=True)
        self.name_dist = name_distance
        self.iter_train = iter_train
        self.output_path = output_path

    def run(self):
        while True:
            item = self.in_q.get()
            if item is None:
                break
            year, self.meta = item

            # Add the president to the members
            self.meta['members'].append({'name': 'Le president'})
            # Set the correct names for the x and y data.
            self.y = [name['name'] for name in self.meta['members']]

            # Set the correct trainingsdata for the names in the parlement.
            self.raw_x = [name.get('abbr') if name.get('abbr') else
                          ' '.join(name['name'].split()[1:]) for name in
                          self.meta['members']]

            # This is for the name distance function
            self.x_y_map = list(zip(self.raw_x, self.y))

            x = self.vectorizer.fit_transform(self.raw_x)
            self.classifier.fit(x, self.y)

            # Load all the texts into the worker.
            texts = []
            for f, info in self.meta['files'].items():
                try:
                    with open('proceedings/procs/'+f) as fle:
                        text = fle.read()
                        texts.append((f, text))
                except:
                    print(f, 'not found')
                    continue

            if self.name_dist:
                x = self.name_distance(texts, max_distance=3)
                self.classifier.fit(x, self.y)

            if self.iter_train:
                x = self.train_on_texts(texts, self.iter_train, k_best=15)
                self.classifier.fit(x, self.y)

            total_parsed = 0
            matched_speeches = 0

            recognized_speakers = set()

            for f, text in texts:
                parsed = []
                file_meta = self.meta['files'][f]

                speeches = re.findall("(.*)—([^—]*\n)", text, re.UNICODE)

                for speech in speeches:
                    classified = self.classify(speech)

                    # Check if the name has been set by the classifier
                    parsed.append(classified)
                    if classified[1] != 'unmatched':
                        matched_speeches += 1
                        recognized_speakers.add(classified[1])

                total_parsed += len(parsed)

                self.create_xml(parsed, file_meta['date'], f,
                                file_meta['pagenum'])


            # Do the validation of the models
            if self.meta['validation']:
                test_x, y_true = zip(*self.meta['validation'])
                self.validate(test_x, y_true, year)

            if texts:
                info = {'num_speeches': total_parsed,
                        'unmatched': total_parsed - matched_speeches,
                        'matched': matched_speeches,
                        'years': year,
                        'recognized speakers': len(recognized_speakers)
                        }
                print(info)

    def validate(self, test_x, y_true, year):
        '''
        Performs validation on the test set which has been set in the Index.
        '''
        correct = 0
        incorrect = 0
        for x in test_x:
            classified = self.classify([x, None], test=False)
            print(year, classified, y_true[test_x.index(x)])
            if y_true[test_x.index(x)].lower() in classified[1].lower():
                correct += 1
            else:
                incorrect += 1
        print('correct', correct, correct / len(test_x) if correct else 0)
        print('incorrect', incorrect, incorrect / len(test_x) if incorrect else 0)

    def rebuild_x_and_y(self, names_speakers, pos_ngrams=False):
        '''
        Rebuilds the raw x and y values for the worker, and gives back a new X
        list which can be transformed and fitted to the classifier.
        '''
        for name, speakers in names_speakers:
            for speaker in speakers:
                self.raw_x.append(speaker)
                self.y.append(name)

        if pos_ngrams:
            return self.ngramvectorizer.transform(self.raw_x)
        return self.vectorizer.transform(self.raw_x)

    def classify(self, speech, test=True):
        '''
        Classifies a speech to a name from the index.
        '''
        speaker = self.create_name(speech[0])
        if speaker: #  or speaker in not_names:

            # classify using the classifier
            test_x = self.vectorizer.transform([speaker])
            classif_name = self.classifier.predict(test_x)[0]
            score = max(self.classifier.predict_log_proba(test_x)[0])

            if test:
                if 1.5 >  len(classif_name) / len(speaker) > 0.4:
                    return score, classif_name, speaker, speech[1]

            return score, classif_name, speaker, speech[1]
        return -9000, 'unmatched', speech[0], speech[1]

    def create_name(self, speaker):
        '''
        The default function for creating names which can be used in the classifier,
        or used in the name_distance function.
        '''
        if ',' in speaker:
            speaker = ''.join(c for word in speaker.split(',')[0] for c in word)

        if '.' in speaker[:4]:
            speaker = ''.join(c for word in speaker.split('.')[1:] for c in word)
        speaker = ''.join(char for char in speaker.strip() if not char.isdigit())
        speaker = speaker.replace('rapporteur', '')

        for not_name in self.meta['not_names']:
            distance = editdistance.eval(speaker, not_name)
            if distance < 6:
                return False

        if len(speaker) > 1 and len(speaker) < 100:
            return speaker
        else:
            return False

    def name_distance(self, texts,  max_distance=3):
        '''
        Create an index of names and occurences in the text with the corresponding edit-distances.
        '''
        best = {name: [] for name in self.y}
        speaker_name = {}

        for f, text in texts:
            speeches = re.findall("(.*)—([^—]*\n)", text, re.UNICODE)

            for speech in speeches:
                speaker = self.create_name(speech[0])

                if speaker:
                    for name_x, name_y in self.x_y_map:
                        distance = editdistance.eval(speaker, name_x)
                        '''
                        if len(speaker) < 6:
                            if distance < 2 and speaker not in speaker_name:
                                best[name_y].append(speaker)
                                if speaker in speaker_name:
                                    speaker_name[speaker].append(name_x)
                                else:
                                    speaker_name[speaker] = [name_x]

                        else:
                        '''
                        if distance < max_distance and speaker not in speaker_name:
                            best[name_y].append(speaker)
                            if speaker in speaker_name:
                                speaker_name[speaker].append(name_x)
                            else:
                                speaker_name[speaker] = [name_x]

        # Remove duplicates to reduce ambiguity in the features.
        '''
        for speaker, names in speaker_name.items():
            if len(names) > 1:
                for name in names:
                    best[name].remove(speaker)
        '''

        return self.rebuild_x_and_y(best.items())

    def train_on_texts(self, texts, num_runs=2, k_best=10, pos_ngrams=False):
        '''
        Iteratively train on the corpus at hand, each time selecting the k-best
        occurrences in the text to be added to the features.
        '''
        best = {name: [] for name in self.y}
        new_y = [n for n in self.y]
        new_x = [n for n in self.y]

        for run in range(num_runs):
            for f, text in texts:
                speeches = re.findall("(.*)—([^—]*\n)", text, re.UNICODE)
                for speech in speeches:
                    classified = self.classify(speech)
                    if classified and classified[1]:
                        heapq.heappush(best[classified[1]], classified[:-1])

            for name, queue in best.items():
                for score, classif_name, speaker in queue[:-k_best:-1]:
                    new_x.append(speaker)
                    new_y.append(classif_name)

            if pos_ngrams:
                x = self.ngramvectorizer.transform(new_x)
            else:
                x = self.vectorizer.transform(new_x)
            self.classifier.fit(x, new_y)

        result = [(name, [name_x[2] for name_x in queue[:-k_best:-1]]) for
                  name, queue in best.items() if queue]

        return self.rebuild_x_and_y(result)

    def vectorize_name(self, name, ngram_range=4):
        positions = []
        for j in range(1, ngram_range + 1):
            for i, ng in enumerate(ngrams(name, j)):
                char_ng = ''.join(ng)
                positions.append(char_ng)
                positions.append(char_ng + str(i))
        return ' '.join(positions)

    def create_xml(self, classified, date, identifier, pagenum):
        n = {"html":"http://www.w3.org/1999/xhtml",
            "pm": "http://www.politicalmashup.nl"}
        nmap = {"html":"{http://www.w3.org/1999/xhtml}",
            "pm": "{http://www.politicalmashup.nl}"}
        # Create root node
        root = etree.Element(nmap["html"] + "html", nsmap=n)

        # Add meta node
        metaElement = etree.Element(nmap["pm"] + "meta", nsmap=n)
        root.append(metaElement)

        # Get date
        dateElement = etree.Element(nmap["pm"] + 'date', nsmap=n)
        dateElement.text = date
        metaElement.append(dateElement)

        # Get identifier
        identifierElement = etree.Element(nmap["pm"] + 'identifier', nsmap=n)
        identifier = "be.proc.ch.d." + identifier.split(".")[0]
        identifierElement.text = identifier
        metaElement.append(identifierElement)

        # Get pages
        pageElement = etree.Element(nmap["pm"] + 'pages', nsmap=n)
        pageElement.text = pagenum
        metaElement.append(pageElement)

        # Get Leglislative period
        legislateElement = etree.Element(nmap["pm"] + 'leglislative_period', nsmap=n)
        legislateElement.text = 'unknown'
        metaElement.append(legislateElement)

        # Session number
        sessionnumberElement = etree.Element(nmap["pm"] + 'session_number', nsmap=n)
        sessionnumberElement.text = '0'
        metaElement.append(sessionnumberElement)

        # House
        houseElement = etree.Element(nmap["pm"] + 'house', nsmap=n)
        houseElement.text = "commons"
        metaElement.append(houseElement)

        # Add proceedings node
        proceedings = etree.Element(nmap["pm"] + "proceedings", nsmap=n)
        root.append(proceedings)

        topic = etree.Element(nmap["pm"] + "topic", nsmap=n)
        topic.set('title', date)
        topic.set('condition', 'topic-like')
        proceedings.append(topic)

        for score, name, raw_name, text in classified:
            if name != 'unmatched':
                member = [member for member in self.meta['members'] if member['name'] == name][0]
            else:
                member = {}
            speechElement = etree.Element(nmap["pm"] + "speech", nsmap=n)
            speechElement.set('condition', 'speech-like')
            speechElement.set('speaker', name)
            speechElement.set('raw_name', raw_name)
            speechElement.set('party', member.get('party', ''))
            speechElement.set('function', 'other')
            speechElement.set('url', member.get('url', ''))
            speechElement.set('district', member.get('district', ''))
            speechElement.set('party-ref', 'be.p.unknown')
            speechElement.text = text
            topic.append(speechElement)

        tree = etree.ElementTree(root)
        tree.write(self.output_path + identifier + ".xml", pretty_print=False, encoding='UTF-16')


def main():
    index = PeriodIndex('mapping.csv', 'periodsV2.json',
                        name_mapping='MPswithmorenames.txt')
    in_q = Queue()
    num_workers = 4

    workers = [Worker(in_q, output_path='./results/') for _ in range(num_workers)]
    for worker in workers:
        worker.start()

    for year, meta in index.index.items():
        if list(year)[-1] < 1910:
            in_q.put((year, meta))

    for _ in range(num_workers):
        in_q.put(None)

    for worker in workers:
        worker.join()

if __name__ == '__main__':
    main()
