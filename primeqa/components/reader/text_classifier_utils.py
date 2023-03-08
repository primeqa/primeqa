from typing import Dict, List

def make_context_for_evc(contextss : List[List[str]],
                         predict_output: Dict,
                         ntoks_left: int,
                         ntoks_right: int,
                         ) -> List[List[str]]:
    contextss_for_evc=[]
    for (example_id, raw_predictions), contexts in zip(predict_output.items(), contextss):
        contexts_for_evc=[]
        for raw_prediction, context in zip(raw_predictions, contexts):
            span_offset=raw_prediction['span_answer']
            b=span_offset['start_position']
            e=span_offset['end_position']
            
            context_for_evc = get_expanded_span_from_offsets(context, b, e, ntoks_left, ntoks_right)
            contexts_for_evc.append(context_for_evc)
        contextss_for_evc.append(contexts_for_evc)
    return contextss_for_evc
    


def get_expanded_span_from_offsets(passage, b, e, ntoks_left, ntoks_right):
    b=max(b,0)
    e=max(e,0)
    b=min(b, len(passage)-1)
    e=min(e, len(passage)-1)
    if b>len(passage)-1:
        return passage
    if e>len(passage)-1:
        e=len(passage)
    print('original span:', passage[b:e], b, e, len(passage))
    nws_left=0
    while b>0 and nws_left < ntoks_left:
        b -= 1
        if passage[b].isspace():
            nws_left += 1

    nws_right=0
    while e<len(passage)-1 and nws_right < ntoks_right:
        e += 1
        if passage[e].isspace():
            nws_right += 1

    print(f'b={b} e={e}')
    if b<e:
        print(passage[b:e])
        return passage[b:e]
    else:
        print('XXXXXXX'+passage)
        return passage


if __name__=='__main__':
    import json
    import IPython
    import pandas as pd

    def make_contexts():
        pred=json.load(open('/home/jsmc/eval_predictions.json'))
        contextss=[]
        questionss=[]
        itemss={}
        spanss=[]
        for uid, items in pred.items():
            contexts=[]
            questions=[]
            spans=[]
            itemsf=[]
            if items[0]['language']=='english':
                for item in items:
                    if item['passage_index']==0:
                        contexts.append(item['passage_answer_text'])
                        questions.append(item['question'])
                        spans.append(item['span_answer_text'])
                        itemsf.append(item)
                if len(contexts)>0:
                    contextss.append(contexts)
                    questionss.append(questions)
                    spanss.append(spans)
                    itemss[uid]=itemsf
        return contextss, questionss, itemss, spanss
                
    contextss, questionss, itemss, spanss = make_contexts()
    contextss_for_evc=make_context_for_evc(contextss, itemss, 15, 15)





