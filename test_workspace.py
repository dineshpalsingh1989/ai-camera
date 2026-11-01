"""Integration checks: python -m unittest test_workspace -v.

Set RUN_TRAINING_SMOKE=1 to include a real one-epoch CPU training job on synthetic test images.
All writes use temporary directories. No live cameras, user credentials or cloud requests are used.
"""
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np
from flask import Flask
from training_service import register_training, safe_extract, TrainingService


def jpg(number, size=160):
    image = np.full((size, size, 3), 30 + number * 7, np.uint8)
    cv2.rectangle(image, (40, 40), (120, 120), (180, 210, 230), -1)
    cv2.putText(image, str(number), (4, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (220, 180, 40), 1)
    return cv2.imencode('.jpg', image)[1].tobytes()


class DummyDetector:
    def __init__(self, path, classes):
        self.path, self.classes = path, classes
        self.key = (path, tuple(classes))


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ctx = {'BASE': str(Path(__file__).parent), 'TRAIN_DIR': str(self.root / 'old-photos'),
                    'settings': {'custom_model': 'previous.pt', 'custom_classes': ['old'],
                                 'cameras': [{'id': 'cam1', 'name': 'Test', 'features': {'custom': False}}]},
                    'cameras': {}, 'CustomDetector': DummyDetector, 'custom_lock': __import__('threading').Lock(),
                    'save_settings': lambda s: None, 'sync_cameras': lambda: None}
        self.app = Flask(__name__)
        self.service = register_training(self.app, self.ctx, self.root / 'workspace')
        self.client = self.app.test_client()

    def tearDown(self):
        for process in list(self.service.processes.values()):
            if process.poll() is None:
                process.terminate(); process.wait(timeout=20)
        self.temp.cleanup()

    def post(self, path, data=None):
        return self.client.post('/api/training' + path, json=data or {})

    def dataset(self, count=6):
        for i in range(count):
            photo = self.service.add_photo(jpg(i), split='train' if i < count-2 else 'val')
            response = self.client.patch('/api/training/photos/' + photo['id'], json={
                'boxes': [{'label':'egg','x1':.25,'y1':.25,'x2':.75,'y2':.75}], 'reviewed':True})
            self.assertEqual(response.status_code, 200, response.json)
        return self.service.build_dataset()

    def wait_job(self, jid, timeout=15):
        end = time.time() + timeout
        while time.time() < end:
            job = self.service.state['jobs'][jid]
            if job['status'] not in {'queued','running','cancelling'}:
                return job
            time.sleep(.1)
        self.fail('Job timed out: ' + repr(job))

    def test_photo_review_validation_and_deduplication(self):
        photo = self.service.add_photo(jpg(0))
        self.assertEqual(photo['id'], self.service.add_photo(jpg(0))['id'])
        image_response = self.client.get('/api/training/photos/'+photo['id']+'/image')
        self.assertEqual(image_response.status_code,200); image_response.close()
        for boxes in [[{'label':'egg','x1':.8,'y1':0,'x2':.1,'y2':1}], [{'label':'egg','x1':float('nan'),'y1':0,'x2':.1,'y2':1}]]:
            self.assertEqual(self.client.patch('/api/training/photos/'+photo['id'],json={'boxes':boxes}).status_code,400)
        self.assertEqual(self.post('/datasets').status_code,400)
        self.assertEqual(self.post('/capture',{'camera':'missing'}).status_code,400)

    def test_dataset_round_trip_and_snapshot_immutability(self):
        dataset = self.dataset()
        self.assertEqual(dataset['counts'], {'train':4,'val':2})
        exported = self.client.get('/api/training/datasets/'+dataset['id']+'/download')
        self.assertEqual(exported.status_code,200)
        data = exported.data; exported.close()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            self.assertNotIn(str(self.root), z.read('data.yaml').decode())
        response=self.client.post('/api/training/datasets',data={'file':(io.BytesIO(data),'dataset.zip')})
        self.assertEqual(response.status_code,200,response.json)
        photo=next(iter(self.service.state['photos'].values()))
        self.client.delete('/api/training/photos/'+photo['id'])
        self.assertTrue(self.service.path(dataset['yaml']).exists())

    def test_zip_traversal_and_executable_exclusion(self):
        for name in ['../outside.txt','C:/outside.txt','..\\outside.txt']:
            data=io.BytesIO()
            with zipfile.ZipFile(data,'w') as z:z.writestr(name,'bad')
            data.seek(0)
            with self.assertRaises(ValueError):safe_extract(data,self.root/'extract')
        data=io.BytesIO()
        with zipfile.ZipFile(data,'w') as z:z.writestr('run.py','raise Exception()')
        data.seek(0);safe_extract(data,self.root/'extract')
        self.assertFalse((self.root/'extract'/'run.py').exists())

    def test_duplicate_images_across_splits_rejected(self):
        dataset=self.dataset()
        folder=self.service.path(dataset['yaml']).parent
        source=next((folder/'train'/'images').glob('*.jpg'))
        dest=next((folder/'val'/'images').glob('*.jpg'))
        dest.write_bytes(source.read_bytes())
        from training_service import prepare_dataset
        with self.assertRaisesRegex(ValueError,'different splits'):prepare_dataset(folder)

    def test_credentials_are_not_returned_and_errors_are_redacted(self):
        self.service.connection={'api_key':'private-test-key','workspace':'test','project':'eggs'}
        self.assertNotIn('private-test-key',self.client.get('/api/training').get_data(as_text=True))
        result=self.service.start('test',lambda jid:(_ for _ in ()).throw(ValueError('bad private-test-key https://x.invalid/?api_key=secret')))
        job=self.wait_job(result['job_id'])
        self.assertEqual(job['status'],'failed');self.assertNotIn('private-test-key',job['message']);self.assertNotIn('secret',job['message'])

    def test_roboflow_upload_includes_reviewed_annotations_and_skips_repeat(self):
        self.service.connection={'api_key':'key','workspace':'test','project':'eggs'}
        photo=self.service.add_photo(jpg(1))
        self.client.patch('/api/training/photos/'+photo['id'],json={'reviewed':True,'boxes':[{'label':'egg','x1':.25,'y1':.25,'x2':.75,'y2':.75}]})
        with patch.object(self.service,'rf',side_effect=[{'success':True,'id':'remote-image'}, {'success':True}]) as remote:
            job=self.post('/roboflow/upload',{'photos':[photo['id']]}).json['job_id']
            self.assertEqual(self.wait_job(job)['status'],'completed')
            self.assertIn('<name>egg</name>',remote.call_args.kwargs['json']['annotationFile'])
            second=self.post('/roboflow/upload',{'photos':[photo['id']]}).json['job_id']
            self.assertEqual(self.wait_job(second)['status'],'completed')
            self.assertEqual(remote.call_count,2)

    def test_roboflow_dataset_download_imports_a_versioned_snapshot(self):
        dataset=self.dataset()
        response=self.client.get('/api/training/datasets/'+dataset['id']+'/download')
        archive=response.data;response.close()
        self.service.connection={'api_key':'key','workspace':'test','project':'eggs'}
        def transfer(jid,url,target,**kwargs):
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive)
        with patch.object(self.service,'rf',return_value={'export':{'link':'https://example.com/dataset.zip'}}), patch.object(self.service,'download',side_effect=transfer):
            jid=self.post('/roboflow/dataset',{'version':2}).json['job_id']
            job=self.wait_job(jid)
            self.assertEqual(job['status'],'completed',job)
            self.assertEqual(job['result']['name'],'eggs v2')
            self.assertEqual(len(self.service.state['datasets']),2)

    def test_remote_download_blocks_private_network_addresses(self):
        with patch('training_service.socket.getaddrinfo',return_value=[(2,1,6,'',('127.0.0.1',443))]):
            with self.assertRaisesRegex(ValueError,'public host'):
                self.service.download('unused','https://example.com/data.zip',self.root/'data.zip')

    def test_cloud_training_default_and_ambiguous_submission_error(self):
        import requests
        with patch.object(self.service,'rf',return_value={'id':'cloud-job'}) as remote:
            response=self.post('/roboflow/train',{'version':1,'epochs':10})
            self.assertEqual(response.status_code,200)
            self.assertEqual(remote.call_args.kwargs['json'],{'epochs':10})
        self.service.connection={'api_key':'key','workspace':'test','project':'eggs'}
        with patch('training_service.requests.request',side_effect=requests.Timeout):
            response=self.post('/roboflow/train',{'version':1,'epochs':10})
            self.assertEqual(response.status_code,400)
            self.assertIn('before submitting again',response.json['error'])

    def test_activation_write_failure_keeps_previous_model(self):
        self.service.state['models']['test']={'id':'test','file':'models/test.pt','classes':['egg'],'evaluations':{'data':{'precision':.8}}}
        def fail(settings):
            raise OSError('disk unavailable')
        self.ctx['save_settings']=fail
        response=self.post('/models/test/activate',{'cameras':['cam1']})
        self.assertEqual(response.status_code,500)
        self.assertEqual(self.ctx['settings']['custom_model'],'previous.pt')
        self.assertEqual(self.service.state['history'],[])

    def test_model_activation_requires_evaluation_and_rolls_back(self):
        self.service.state['models']['test']={'id':'test','file':'models/test.pt','classes':['egg'],'evaluations':{}}
        self.assertEqual(self.post('/models/test/activate',{'cameras':['cam1']}).status_code,400)
        self.service.state['models']['test']['evaluations']={'data':{'precision':.8}}
        self.assertEqual(self.post('/models/test/activate',{'cameras':['cam1']}).status_code,200)
        self.assertEqual(self.ctx['settings']['custom_classes'],['egg'])
        self.assertTrue(self.ctx['settings']['cameras'][0]['features']['custom'])
        self.assertEqual(self.post('/rollback').status_code,200)
        self.assertEqual(self.ctx['settings']['custom_model'],'previous.pt')
        self.assertFalse(self.ctx['settings']['cameras'][0]['features']['custom'])

    def test_job_cancel_and_restart_recovery(self):
        def work(jid):
            for _ in range(100):
                self.service.check_cancel(jid);time.sleep(.02)
        jid=self.service.start('wait',work)['job_id']
        self.assertEqual(self.post('/jobs/'+jid+'/cancel').status_code,200)
        self.assertEqual(self.wait_job(jid)['status'],'cancelled')
        self.service.state['jobs']['old']={'id':'old','status':'running','created':0};self.service.save()
        restarted=TrainingService(self.ctx,self.service.root)
        self.assertEqual(restarted.state['jobs']['old']['status'],'interrupted')

    def test_sampling_is_opt_in_and_requires_review(self):
        frame=cv2.imdecode(np.frombuffer(jpg(2),np.uint8),cv2.IMREAD_COLOR)
        detections=[('egg',.6,40,40,120,120)]
        self.service.sample('cam1',frame,detections)
        self.assertFalse(self.service.state['photos'])
        self.assertEqual(self.post('/auto-collect',{'camera':'cam1','enabled':True,'interval':10,'confidence':.7}).status_code,200)
        self.service.sample('cam1',frame,detections)
        self.assertEqual(len(self.service.state['photos']),1)
        photo=next(iter(self.service.state['photos'].values()));self.assertFalse(photo['reviewed'])
        self.service.last_sample.clear();self.service.sample('cam1',frame,detections)
        self.assertEqual(len(self.service.state['photos']),1)

    @unittest.skipUnless(os.environ.get('RUN_TRAINING_SMOKE')=='1','Enable RUN_TRAINING_SMOKE for real CPU training')
    def test_real_one_epoch_training_and_validation(self):
        dataset=self.dataset()
        response=self.post('/train',{'dataset':dataset['id'],'epochs':1,'imgsz':160,'batch':2,'device':'cpu'})
        self.assertEqual(response.status_code,202,response.json)
        job=self.wait_job(response.json['job_id'],timeout=240)
        if job['status']!='completed':
            log=self.service.path('jobs/'+job['id']+'/worker.log')
            print(log.read_text(encoding='utf-8')[-7000:] if log.exists() else '')
        self.assertEqual(job['status'],'completed',job)
        model=self.service.state['models'][job['result']['model_id']]
        self.assertEqual(model['classes'],['egg'])
        self.assertTrue(self.service.path(model['file']).is_file())
        self.assertIn('precision',model['evaluations'][dataset['id']])


if __name__=='__main__':
    unittest.main()
