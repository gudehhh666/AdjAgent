# Harness test fixtures

`protein_example.fasta` is the exact 18-residue sequence selected for the explicitly new protein extensibility evaluation. The test verifies parsing and calculations, not a biological function.

`structure_example.pdb` is a deliberately synthetic parser fixture: one model, one chain, two residues and four atoms. Its coordinates are not an experimental or predicted protein structure and must never be cited as scientific evidence.

`mcp_echo_server.py` is an actual local MCP SDK server for protocol/lifecycle tests, not an external scientific backend. It is launched only with explicit host test configuration.
