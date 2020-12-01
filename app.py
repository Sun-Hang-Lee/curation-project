from flask import Flask, Response
from flask_restplus import Api, Resource
from src.curation_main import CurationMain
import json

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
api = Api(app, version='1.0', title='튜터 추천 API', description='튜터 추천 조회 API입니다')
ns = api.namespace('curation', description='튜터 추천 조회')

curation_parser = ns.parser()
curation_parser.add_argument('user', required=True, help='학생 ID')
curation_parser.add_argument('recommend_count', required=True, help='튜터 추천 수(과제 조건은 3)')


@ns.route('/curation')
@ns.expect(curation_parser)
class Curation(Resource):
    def get(self):
        args = curation_parser.parse_args()
        try:
            user = int(args['user'])
            recommend_count = int(args['recommend_count'])
            cm = CurationMain()
            result = cm.main_process(user, recommend_count)
            res = json.dumps(result, ensure_ascii=False).encode('utf-8')
            return Response(res, content_type='application.json; charset=utf-8')
        except KeyError:
            return {'result': 'ERROR_PARAMETER'}, 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)


