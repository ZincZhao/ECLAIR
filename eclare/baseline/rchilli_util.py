# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-06-04 17:46
import math
from datetime import datetime


def input_from_rchilli(parsed_data: dict, all_sections=False):
    sections = []
    output = parsed_data
    output.update({'id': parsed_data['ResumeFileName'].split('.')[0], 'sections': sections})
    education = []
    for each in parsed_data['SegregatedQualification']:
        education.append(each['Degree']['DegreeName'])
    sections.append({'type': 'Education', 'content': '\n'.join(education)})
    experience = []
    for each in parsed_data['SegregatedExperience']:
        info = [each['Employer']['EmployerName'], each['JobProfile']['Title']]
        try:
            start_date = each['StartDate']
            end_date = each['EndDate']
            years = date_to_year(start_date, end_date)
            # months = (end - start).days % 365 // 30
            info.append(f'{years} years')
            # info.append(f'{years:.1f} years')
        except:
            pass
        experience.append(', '.join(info))
    sections.append({'type': 'Work Experience', 'content': '\n'.join(experience)})
    if all_sections:
        sections.append({'type': 'Profile', 'content': ' '.join(set(str(x) for x in parsed_data['Name'].values()))})
        sections.append({'type': 'Skills', 'content': parsed_data['SkillBlock']})
        sections.append({'type': 'Other', 'content': parsed_data['DetailResume']})
        sections.append({'type': 'Activities', 'content': parsed_data['Summary']})
    return output


def date_to_year(start_date, end_date):
    if not start_date and not end_date:
        return 1
    try:
        start = datetime.strptime(start_date, '%d/%m/%Y')
        end = datetime.strptime(end_date, '%d/%m/%Y')
        years = math.ceil((end - start).days / 365)
        return years
    except:
        return 0
