import asyncio
import pytest
from adj_partner.profiles import PROFILES

def test_new_session_default_and_explicit_profile_survive_reopen(tmp_path):
    from adj_partner.application import AgentService
    from adj_partner.cli import parser
    from adj_partner.models import ScriptedPort
    assert parser().parse_args(['new']).profile == 'research-evidence-bound'
    service = AgentService(tmp_path/'state', model=ScriptedPort([]))
    try:
        fresh = service.create_session()
        legacy = service.create_session(profile='general')
        assert fresh['config']['profile'] == 'research-evidence-bound'
        assert 'ask_user' in fresh['config']['enabled_tools']
        assert fresh['config']['allow_network'] is False
        assert fresh['config']['allowed_roots'] == []
    finally:
        service.close()
    service = AgentService(tmp_path/'state', model=ScriptedPort([]))
    try:
        assert service.session(fresh['id'])['config']['profile'] == 'research-evidence-bound'
        assert service.session(legacy['id'])['config']['profile'] == 'general'
    finally:
        service.close()

def test_tui_initial_session_and_new_command_use_default(tmp_path):
    from adj_partner.application import AgentService
    from adj_partner.models import ScriptedPort
    from adj_partner.tui import AdjPartnerApp
    service = AgentService(tmp_path/'state', model=ScriptedPort([]))
    app = AdjPartnerApp(service)
    async def scenario():
        async with app.run_test(size=(100,30)):
            initial = app.session_id
            assert service.session(initial)['config']['profile'] == 'research-evidence-bound'
            app._command('/new')
            assert app.session_id != initial
            assert service.session(app.session_id)['config']['profile'] == 'research-evidence-bound'
            app._command('/new general')
            assert service.session(app.session_id)['config']['profile'] == 'general'
    try:
        asyncio.run(scenario())
    finally:
        service.close()

@pytest.mark.parametrize('name',list(PROFILES))
def test_cli_uses_authoritative_profile_registry(name):
    from adj_partner.cli import parser
    assert parser().parse_args(['new','--profile',name]).profile==name

def test_tui_new_accepts_optional_profiles_without_network_grant(tmp_path):
    from adj_partner.application import AgentService
    from adj_partner.models import ScriptedPort
    from adj_partner.tui import AdjPartnerApp,HELP
    service=AgentService(tmp_path/'state',model=ScriptedPort([]))
    sid=service.create_session()['id'];app=AdjPartnerApp(service,sid)
    async def scenario():
        async with app.run_test(size=(100,30)):
            for name in ['adjuvant-resource','adjuvant-grounded','adjuvant-batch','adjuvant-routed']:
                assert name in HELP
                app._command('/new '+name)
                config=service.session(app.session_id)['config']
                assert config['profile']==name and config['allow_network'] is False
    try:asyncio.run(scenario())
    finally:service.close()
