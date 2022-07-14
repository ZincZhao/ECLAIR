import json


class duplication_extractor:
    def extract(s: str) -> dict:
        if not s.endswith('.json'):
            return None
        f = json.load(open(s))
        f = f['ResumeParserData']
        q = f['Qualification']
        c = f['Certification']
        e = f['Experience']
        jp = f['JobProfile']
        res = {'Qualification': q, 'Certification': c, 'Experience': e, 'JobProfile': jp}
        return res
