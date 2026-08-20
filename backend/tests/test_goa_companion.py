import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import process_rag_pipeline, resources

@pytest.fixture(scope="module", autouse=True)
def setup_resources():
    if not resources.ready:
        resources.initialize()

def test_goa_beaches():
    res = process_rag_pipeline("Best beaches in Goa?")
    assert res["status"] == "answered"
    assert res["grounded"] is True
    assert res["category"] == "Beaches"
    ans = res["answer"].lower()
    assert any(b in ans for b in ["anjuna", "vagator", "palolem", "baga", "calangute", "agonda"])
    assert len(res["sources"]) > 0

def test_goa_2day_itinerary():
    res = process_rag_pipeline("Plan a 2 day Goa trip.")
    assert res["status"] == "answered"
    assert res["grounded"] is True
    assert res["category"] == "Itinerary"
    ans = res["answer"].lower()
    assert any(w in ans for w in ["day 1", "day 2", "day 3", "north goa", "south goa", "aguada", "palolem", "beach", "trip"])
    assert len(res["sources"]) > 0

def test_goa_old_goa():
    res = process_rag_pipeline("What should I visit in Old Goa?")
    assert res["status"] == "answered"
    assert res["grounded"] is True
    assert res["category"] == "Heritage"
    ans = res["answer"].lower()
    assert ("bom jesus" in ans or "se cathedral" in ans or "st. francis" in ans)

def test_goa_food():
    res = process_rag_pipeline("What Goan food should I try?")
    assert res["status"] == "answered"
    assert res["grounded"] is True
    assert res["category"] == "Food"
    ans = res["answer"].lower()
    assert ("fish curry" in ans or "poi" in ans or "bebinca" in ans or "vindaloo" in ans or "xacuti" in ans)

def test_goa_sunset():
    res = process_rag_pipeline("Best places for sunset in Goa?")
    assert res["status"] == "answered"
    assert res["grounded"] is True
    assert res["category"] in ["Heritage", "General Goa", "Beaches"]
    ans = res["answer"].lower()
    assert ("chapora" in ans or "aguada" in ans or "palolem" in ans or "vagator" in ans or "cabo de rama" in ans)

def test_goa_culture():
    res = process_rag_pipeline("Tell me about Goa's culture.")
    assert res["status"] == "answered"
    assert res["grounded"] is True
    assert res["category"] == "Culture"
    ans = res["answer"].lower()
    assert ("carnival" in ans or "shigmo" in ans or "konkani" in ans or "portuguese" in ans or "sao joao" in ans)

def test_goa_family():
    res = process_rag_pipeline("Which places are good for families in Goa?")
    assert res["status"] == "answered"
    assert res["grounded"] is True
    assert res["category"] == "Family"
    ans = res["answer"].lower()
    assert ("dudhsagar" in ans or "spice" in ans or "miramar" in ans or "palolem" in ans or "science centre" in ans)

def test_goa_relaxed_itinerary():
    res = process_rag_pipeline("Give me a relaxed itinerary for Goa.")
    assert res["status"] == "answered"
    assert res["grounded"] is True
    assert res["category"] == "Itinerary"
    ans = res["answer"].lower()
    assert ("agonda" in ans or "palolem" in ans or "south goa" in ans or "butterfly" in ans)

def test_goa_south_goa():
    res = process_rag_pipeline("Best places to explore in South Goa?")
    assert res["status"] == "answered"
    assert res["grounded"] is True
    ans = res["answer"].lower()
    assert ("palolem" in ans or "agonda" in ans or "cabo de rama" in ans or "cotigao" in ans or "chandor" in ans)

def test_goa_panjim():
    res = process_rag_pipeline("What can I do near Panjim?")
    assert res["status"] == "answered"
    assert res["grounded"] is True
    ans = res["answer"].lower()
    assert ("fontainhas" in ans or "immaculate conception" in ans or "mandovi" in ans or "miramar" in ans)

def test_goa_conversational_followup():
    turn1_query = "Best beaches in Goa?"
    turn1_res = process_rag_pipeline(turn1_query)
    assert turn1_res["status"] == "answered"

    history = [
        {"role": "user", "content": turn1_query},
        {"role": "assistant", "content": turn1_res["answer"]}
    ]

    turn2_query = "Which one is peaceful?"
    turn2_res = process_rag_pipeline(turn2_query, history=history)
    assert turn2_res["status"] == "answered"
    assert turn2_res["grounded"] is True
    ans = turn2_res["answer"].lower()
    assert ("agonda" in ans or "palolem" in ans or "south goa" in ans or "morjim" in ans or "butterfly" in ans or "mandrem" in ans or "peaceful" in ans)

def test_goa_multilingual_queries():
    # Hindi
    res_hi = process_rag_pipeline("गोवा में 2 दिन का ट्रिप कैसे प्लान करें?")
    assert res_hi["status"] == "answered"
    assert res_hi["grounded"] is True
    assert len(res_hi["sources"]) > 0

    # Marathi / Konkani
    res_mr = process_rag_pipeline("गोवा राज्याची राजधानी कोणती?")
    assert res_mr["status"] == "answered"
    assert res_mr["grounded"] is True
    assert any(w in res_mr["answer"].lower() for w in ["पणजी", "panaji", "panjim"])

def test_unsupported_refusals():
    # Unsupported query must be refused cleanly
    res1 = process_rag_pipeline("How many aliens visited Goa in 1850?")
    assert res1["status"] == "refused"
    assert res1["grounded"] is False
    assert res1["context_sufficient"] is False

    res2 = process_rag_pipeline("What is the recipe for chocolate lava cake?")
    assert res2["status"] == "refused"
    assert res2["grounded"] is False
