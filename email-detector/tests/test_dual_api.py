import asyncio
import json

import httpx

import api


def test_embeddings_and_chat_use_separate_hosts_and_credentials(monkeypatch):
    monkeypatch.setattr(api, 'API_URL', 'https://chat.invalid/v1')
    monkeypatch.setattr(api, 'API_KEY', 'cloud-test-token')
    monkeypatch.setattr(api, 'EMBED_API_URL', 'http://127.0.0.1:1234/v1')
    monkeypatch.setattr(api, 'EMBED_API_KEY', 'local-test-token')
    seen = []

    def handler(request):
        seen.append(request.url.path)
        if request.url.path.endswith('/embeddings'):
            assert request.url.host == '127.0.0.1'
            assert request.headers['authorization'] == 'Bearer local-test-token'
            return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [3., 4.]}]})
        assert request.url.host == 'chat.invalid'
        assert request.headers['authorization'] == 'Bearer cloud-test-token'
        return httpx.Response(200, json={'choices': [{'message': {
            'content': json.dumps([{'i': 0, 'score': 0.7}])}}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await asyncio.gather(api.embedding_many(['sample'], client=client),
                                        api.semantic_many([('body', 'sample')], client=client))
    embedding, semantic = asyncio.run(run())
    assert embedding == [[0.6, 0.8]]
    assert semantic == [0.7]
    assert len(seen) == 2


def test_embedding_cache_tracks_its_own_endpoint(monkeypatch):
    monkeypatch.setattr(api, 'EMBED_API_URL', 'http://127.0.0.1:1234/v1')
    before = api.signal_key('', 'sample', 'embedding')
    monkeypatch.setattr(api, 'API_URL', 'https://another-chat.invalid/v1')
    assert api.signal_key('', 'sample', 'embedding') == before
    monkeypatch.setattr(api, 'EMBED_API_URL', 'http://127.0.0.1:9999/v1')
    assert api.signal_key('', 'sample', 'embedding') != before
