from mongoengine import connect, Document, StringField, IntField, MapField
from tqdm import tqdm

connect('testdb')

class MongoDict(Document):
    uid = StringField(required=True, primary_key=True)
    wordfreqs = MapField(field=IntField(), required=True)

    @classmethod
    def __getitem__(cls, key):
        pass
    
    @classmethod
    def __setitem__(cls, key, value):
        cls(uid=)

    @classmethod
    def __contains__(cls, key):
        pass

    @classmethod
    def __missing__(cls, key):
        pass