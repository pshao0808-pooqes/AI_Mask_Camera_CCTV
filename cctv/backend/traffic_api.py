import requests
import xml.etree.ElementTree as ET
from datetime import datetime

class TrafficAccidentAPI:
    def __init__(self):
        self.api_key = "YOUR_API_KEY_HERE"  # 발급받은 키로 교체
        self.base_url = "https://opendata.koroad.or.kr/api/rest"
        
    def get_accident_data(self, sido_code="11", gugun_code="", year="2023"):
        """
        사고다발지역 데이터 조회
        sido_code: 11(서울), 42(강원) 등
        """
        url = f"{self.base_url}/AccidentMultiSpot"
        params = {
            'authKey': self.api_key,
            'searchYearCd': year,
            'siDo': sido_code,
            'guGun': gugun_code,
            'type': 'xml',
            'numOfRows': 50,
            'pageNo': 1
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return self._parse_xml_response(response.content)
        except Exception as e:
            print(f"API 호출 실패: {e}")
            
        # API 실패시 샘플 데이터 반환
        return self._get_sample_data()
    
    def _parse_xml_response(self, xml_content):
        root = ET.fromstring(xml_content)
        accidents = []
        
        for item in root.findall('.//item'):
            accident = {
                'id': item.findtext('afos_id', ''),
                'spot_name': item.findtext('spot_nm', ''),
                'address': item.findtext('sido_sgg_nm', ''),
                'lat': float(item.findtext('la_crd', '37.5665')),
                'lon': float(item.findtext('lo_crd', '126.9780')),
                'accident_count': int(item.findtext('occrrnc_cnt', '0')),
                'death_count': int(item.findtext('dth_dnv_cnt', '0')),
                'injury_count': int(item.findtext('caslt_cnt', '0')),
                'severe_injury': int(item.findtext('se_dnv_cnt', '0')),
                'light_injury': int(item.findtext('sl_dnv_cnt', '0'))
            }
            
            # 심각도 계산
            if accident['death_count'] > 0:
                accident['severity'] = '사망'
            elif accident['severe_injury'] > 0:
                accident['severity'] = '중상'
            else:
                accident['severity'] = '경상'
                
            accidents.append(accident)
            
        return accidents
    
    def _get_sample_data(self):
        return [
            {
                'id': 'sample_1',
                'spot_name': '강남대로 교보타워 앞',
                'address': '서울특별시 강남구',
                'lat': 37.5013, 'lon': 127.0373,
                'accident_count': 15, 'death_count': 1,
                'injury_count': 8, 'severity': '중상'
            },
            {
                'id': 'sample_2',
                'spot_name': '종로 광화문 사거리', 
                'address': '서울특별시 종로구',
                'lat': 37.5759, 'lon': 126.9768,
                'accident_count': 12, 'death_count': 0,
                'injury_count': 6, 'severity': '경상'
            }
        ]