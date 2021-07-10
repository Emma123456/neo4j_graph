from py2neo import Graph
import numpy as np
import pandas as pd

graph = Graph("http://127.0.0.1:7474", auth=('neo4j', "cuixy123"))


def create_attribute_question():
    company = graph.run('MATCH (n:company) RETURN n.name as name').to_ndarray()
    person = graph.run('MATCH (n:person) RETURN n.name as name').to_ndarray()

    questions = []

    for c in company:
        c = c[0].strip()
        question = f"{c}的收益"
        questions.append(question)
        question = f"{c}的收入"
        questions.append(question)

    for p in person:
        p = p[0].strip()
        questions.append(f"{p}的年龄是几岁")
        questions.append(f"{p}多大")
        questions.append(f"{p}几岁")
        questions.append(f"{p}的年龄")
        questions.append(f"{p}的年纪")

    return questions


def create_entity_question():
    questions = []
    for _ in range(250):
        for op in ['大于', '等于', '小于', '是', '有']:
            profit = np.random.randint(10000, 10000000, 1)[0]
            question = f"收益{op}{profit}的公司有哪些"
            questions.append(question)
            profit = np.random.randint(10000, 10000000, 1)[0]
            question = f"哪些公司收益{op}{profit}"
            questions.append(question)

    for _ in range(250):
        for op in ['大于', '等于', '小于', '是', '有']:
            age = np.random.randint(20, 70, 1)[0]
            question = f"年龄{op}{age}的人有哪些"
            questions.append(question)
            age = np.random.randint(20, 70, 1)[0]
            question = f"哪些人年龄{op}{age}"
            questions.append(question)

    company = graph.run('MATCH (n:company) RETURN n.name as name').to_ndarray()
    person = graph.run('MATCH (n:person) RETURN n.name as name').to_ndarray()
    for c in company:
        c = c[0].strip()
        questions.append(c)
    for p in person:
        p = p[0].strip()
        questions.append(p)

    return questions


def create_relation_question():
    relation = graph.run('MATCH (n)-[r]->(m) RETURN n.name as name, type(r) as r').to_ndarray()

    questions = []

    for r in relation:
        if str(r[1]) in ['董事', '监事']:
            question = f"{r[0]}的{r[1]}是谁"
            question = f"{r[0]}的{r[1]}"
            questions.append(question)
        else:
            questions.append(f"{r[0]}的{r[1]}")
            questions.append(f"{r[0]}的{r[1]}是啥")
            questions.append(f"{r[0]}的{r[1]}什么")
    return questions


q1 = create_entity_question()
q2 = create_attribute_question()
q3 = create_relation_question()

df = pd.DataFrame()
df["question"] = q1 + q2 + q3
df["label"] = [0] * len(q1) + [1] * len(q2) + [2] * len(q3)

df.to_csv('question_classification.csv', encoding='utf_8_sig', index=False)


