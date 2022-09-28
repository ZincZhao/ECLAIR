import json


class duplication_extractor_pretrain:
    def extract(s: str) -> dict:
        if not s.endswith('.json'):
            return None
        f = json.load(open(s))
        f = f['ResumeParserData']
        q = f['Qualification']
        c = f['Certification']
        e = f['Experience']
        jp = f['JobProfile']
        # res = {'Qualification': q, 'Certification': c, 'Experience': e, 'JobProfile': jp}
        res = q + ' ' + c + ' ' + e + ' ' + jp
        res = res.replace("\r", "")
        res = res.replace("\t", "")
        res = res.replace(".", ".#")
        res = res.replace("?", "?#")
        res = res.replace("!", "!#")
        res = res.split("#")
        return res
