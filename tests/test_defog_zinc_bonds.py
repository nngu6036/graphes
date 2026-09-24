import networkx as nx
import pytest

from grapher.models.defog.workers.prepare_molecular_dataset import _zinc_kekule_edges


Chem = pytest.importorskip("rdkit.Chem")


def make_graph(smiles, *, kekule):
    mol = Chem.MolFromSmiles(smiles)
    if kekule:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    graph = nx.Graph(source_smiles=smiles)
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum())
    for bond in mol.GetBonds():
        graph.add_edge(
            bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
            bond_type=4 if bond.GetIsAromatic() else int(bond.GetBondTypeAsDouble()),
        )
    return graph


def convert(graph):
    nodes = list(graph)
    return _zinc_kekule_edges(
        graph, nodes=nodes,
        node_positions={node: i for i, node in enumerate(nodes)},
        graph_label="train[0]",
    )


@pytest.mark.parametrize("smiles", ["c1ccccc1", "CCOc1ccc(Cl)cc1", "c1ccc2ccccc2c1", "CCO"])
def test_prepared_kekule_bonds_are_preserved(smiles):
    graph = make_graph(smiles, kekule=True)
    edges, record = convert(graph)
    assert edges == {tuple(sorted((u, v))): d["bond_type"] for u, v, d in graph.edges(data=True)}
    assert record["kekulized"] is False


def test_legacy_aromatic_bonds_are_converted():
    graph = make_graph("c1ccccc1", kekule=False)
    edges, record = convert(graph)
    assert sorted(edges.values()) == [1, 1, 1, 2, 2, 2]
    assert record["aromatic_bonds_input"] == 6
    assert record["kekulized"] is True


def test_incorrect_bond_order_is_rejected():
    graph = make_graph("CCO", kekule=True)
    graph.edges[0, 1]["bond_type"] = 2
    with pytest.raises(ValueError, match="bonds do not exactly reproduce"):
        convert(graph)
