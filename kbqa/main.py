import torch
from kbqa.data_process import *
import ahocorasick
from py2neo import Graph
import re
import traceback

model = torch.load('text_cnn.p')
model.eval()

graph = Graph('http；//127.0.0.1:7474', auth=('neo4j','cuixy123'))
company = graph.run('MATCH (n:company) RETURN n.name as name').to_ndarray()
person = graph.run('MATCH (n:person) RETURN n.name as name').to_ndarray()
relation = graph.run('MATCH ()-[r]-() RETURN distinct type(r)').to_ndarray()

ac_company = ahocorasick.Automaton()
ac_person = ahocorasick.Automaton()
ac_relation = ahocorasick.Automaton()

for key in enumerate(company):
    ac_company.add_word(key[1][0], key[1][0])
for key in enumerate(person):
    ac_person.add_word(key[1][0], key[1][0])
for key in enumerate(relation):
    ac_relation.add_word(key[1][0], key[1][0])
ac_relation.add_word('年龄', '年龄')
ac_relation.add_word('年纪', '年纪')
ac_relation.add_word('收入', '收入')
ac_relation.add_word('收益', '收益')

ac_company.make_automaton()
ac_person.make_automaton()
ac_relation.make_automaton()
device = torch.device(f'cuda:{0}') if torch.cuda.is_available() else torch.device('cpu')


def classification_predict(s):
    s = seq2index(s)
    s = torch.from_numpy(padding_seq([s])).to(device).long()
    out = model(s)
    out = out.cpu().data.numpy()
    print(out)
    return out.argmax(1)[0]


def entity_link(text):
    subject = []
    subject_type = None
    for end_index, original_value in ac_company.iter(text):
        start_index = end_index - len(original_value) + 1
        # print('实体', (start_index, end_index, original_value))
        assert text[start_index: start_index + len(original_value)] == original_value
        subject.append(original_value)
        subject_type = 'company'
    for end_index, original_value in ac_person.iter(text):
        start_index = end_index - len(original_value) + 1
        # print('实体', (start_index, end_index, original_value))
        assert text[start_index:start_index + len(original_value)] == original_value
        subject.append(original_value)
        subject_type = 'person'
    if len(subject) > 0:
        return subject[0], subject_type
    else:
        return None, subject_type


def get_op(text):
    pattern = re.compile(r'\d+')
    num = pattern.findall(text)
    op = None
    if '大于' in text:
        op = '>'
    elif '小于' in text:
        op = '<'
    elif '等于' in text or '是' in text:
        op = '='
    return op, float(num[0]) if len(num) > 0 and len(num[0]) > 0 else None


def kbqa(text):
    print('*' * 100)
    cls = classification_predict(text)
    print('question type:', cls)
    res = ''
    subject, subject_type = entity_link(text)

    if cls == 0:
        if subject_type is not None:
            cypher = f'match (n:{subject_type}) where n.name = "{subject}" return n'
            # print(cypher)
            res = graph.run(cypher).to_ndarray()
        else:
            op, num = get_op(text)
            if op is not None:
                subject_type = ''
                attribute = ''
                for w in ['年龄', '年纪']:
                    if w in text:
                        subject_type = 'person'
                        attribute = 'age'
                        break
                for w in ['收入', '收益']:
                    if w in text:
                        subject_type = 'company'
                        attribute = 'profit'
                        break
                cypher = f'match (n:{subject_type}) where n.{attribute}{op}{num} return n.name'
                # print(cypher)
                res = graph.run(cypher).to_ndarray()
    elif cls == 1:
        predicate = ''
        for w in ['年龄', '年纪', '几岁', '多大']:
            if w in text and subject_type == 'person':
                predicate = 'age'
                break
        for w in ['收入', '收益']:
            if w in text and subject_type == 'company':
                predicate = 'profit'
                break
        cypher = f'match (n:{subject_type}) where n.name = "{subject}" return n.{predicate}'
        # print(cypher)
        res = graph.run(cypher).to_ndarray()
    elif cls == 2:
        subject = ''
        subject_index_dict = {}
        for end_index, original_value in ac_company.iter(text):
            start_index = end_index - len(original_value) + 1
            # print('实体', (start_index, end_index, original_value))
            assert text[start_index: start_index + len(original_value)] == original_value
            subject = original_value
            subject_index_dict[start_index] = end_index
        predicate = []
        for end_index, original_value in ac_relation.iter(text):
            start_index = end_index - len(original_value) + 1
            valid = True
            for subject_s, subject_end in subject_index_dict.items():
                if subject_s <= start_index <= subject_end:
                    valid = False
                    break
            if valid:
                # print('关系', (start_index, end_index, original_value))
                assert text[start_index: start_index + len(original_value)] == original_value
                predicate.append(original_value)
        for i, p in enumerate(predicate):
            cypher = f"match (s:company)-[p:`{p}`]->(o) where s.name='{subject}' return o.name"
            # print(cypher)
            res = graph.run(cypher).to_ndarray()
            if len(res) == 0:
                break
            subject = res[0][0]
            if i == len(predicate) - 1:
                break
            new_index = text.index(p) + len(p)
            new_question = subject + str(text[new_index:])
            print('new question:', new_question)
            res = kbqa(new_question)
            break
    return res


if __name__ == '__main__':
    while 1:
        try:
            text = input('text:')
            res = kbqa(text)
            print(res)
        except:
            print(traceback.format_exc())






