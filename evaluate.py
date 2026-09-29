import json
import sqlite3
import os

from typesafe_sdk import Noul, Choice, Score, TypeSafeClient
from confidence_analysis import compute_confidence
from calibration import compute_calibration

DB_PATH = os.path.join(os.path.dirname(__file__), "jev_eval.db")
EXAMPLES_PATH = os.path.join(os.path.dirname(__file__), "labeled_examples.json")


def setup_database(db_path=DB_PATH):
    """Create the results table with confidence column."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS results
                 (id INTEGER PRIMARY KEY,
                  state TEXT,
                  question TEXT,
                  jev_prob REAL,
                  confidence REAL,
                  choice TEXT,
                  choice_probs TEXT,
                  score_value REAL,
                  score_conf REAL,
                  score_probs TEXT,
                  ground_truth INTEGER,
                  correct INTEGER,
                  noul_correct INTEGER)''')
    conn.commit()
    return conn


def load_examples(path=EXAMPLES_PATH):
    """Load labeled examples from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(examples, conn):
    """Run each example through Jev and store results."""
    c = conn.cursor()
    client = TypeSafeClient(
    base_url="https://api.typesafe.pro",
    api_key="anonymous"
)
    
    results = []

    total_input_tokens = 0
    total_output_tokens = 0
    
    for ex in examples:
        try:
            # Noul question: returns a probability
            noul_response = client.system_one(
                state=ex["state"],
                questions={"q": Noul(instructions=ex["question"])}
            )
            jev_prob = noul_response.nouls["q"].noul
            
            # Choice question: returns choice + confidence
            choice_response = client.system_one(
                state=ex["state"],
                questions={
                    "category": Choice(
                        instructions=ex["question"],
                        criteria={
                            "yes": "The answer to the question is yes",
                            "no": "The answer to the question is no"
                        }
                    )
                }
            )
            choice = choice_response.choices["category"].choice
            confidence = choice_response.choices["category"].confidence
            choice_probs = choice_response.choices["category"].probabilities
            # TEMP DEBUG — inspect the Choice object once
            if not hasattr(run_evaluation, "_printed"):
                obj = choice_response.choices["category"]
                #print("CHOICE OBJ TYPE:", type(obj))
                #print("CHOICE ATTRS:",
                [a for a in dir(obj) if not a.startswith("_")]
                #print("CHOICE __dict__:", getattr(obj, "__dict__", None))
                run_evaluation._printed = True
            # TEMP: probe Score head
                        # Score question: ordinal rating with full distribution
            score_response = client.system_one(
                state=ex["state"],
                questions={
                    "rating": Score(
                        instructions=ex["question"],
                        criteria=[
                            "Not at all - completely unrelated or not applicable",
                            "Slightly - tangentially related",
                            "Moderately - clearly related but not strongly",
                            "Strongly - directly on topic",
                            "Very strongly - the central subject of the message",
                        ],
                    )
                }
            )
            score_value = score_response.scores["rating"].score
            score_conf = score_response.scores["rating"].confidence
            score_probs = score_response.scores["rating"].probabilities
        
            if not hasattr(run_evaluation, "_score_printed"):
                obj = score_response.scores["rating"]
                #print("SCORE OBJ TYPE:", type(obj))
                #print("SCORE ATTRS:", [a for a in dir(obj) if not a.startswith("_")])
                #print("SCORE __dict__:", getattr(obj, "__dict__", None))
                run_evaluation._score_printed = True
            input_tokens = choice_response.usage.input_tokens
            output_tokens = choice_response.usage.output_tokens

            #print(f"[{ex['id']}] Input tokens: {input_tokens}, Output tokens: {output_tokens}")

        # Accumulate for the final total
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            
            # Determine correctness
            # Noul: correct if prob >= 0.5 matches ground truth
            noul_correct = int((jev_prob >= 0.5) == bool(ex["ground_truth"]))
            # Choice: correct if choice matches ground truth
            choice_correct = int((choice == "yes") == bool(ex["ground_truth"]))
            
            #c.execute(
            #   """INSERT OR REPLACE INTO results
             #      (id, state, question, jev_prob, confidence, choice,  ground_truth, correct)
              #     VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
               # (ex["id"], ex["state"], ex["question"],
                # jev_prob, confidence, choice,  ex["ground_truth"], choice_correct)
            #)
            c.execute(
                """INSERT OR REPLACE INTO results
                   (id, state, question, jev_prob, confidence, choice,
                    choice_probs, score_value, score_conf, score_probs,
                    ground_truth, correct, noul_correct)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ex["id"], ex["state"], ex["question"],
                 jev_prob, confidence, choice,
                 json.dumps(choice_probs),
                 score_value, score_conf, json.dumps(score_probs),
                 ex["ground_truth"], choice_correct, noul_correct)
            )
            results.append({
                "id": ex["id"],
                "jev_prob": jev_prob,
                "confidence": confidence,
                "choice": choice,
                "choice_probs": choice_probs,
                "ground_truth": ex["ground_truth"],
                "noul_correct": noul_correct,
                "choice_correct": choice_correct,
                "score_value": score_value,
                "score_conf": score_conf,
                "score_probs": score_probs,
                
            })
            
            if (len(results)) % 50 == 0:
                print(f"  Processed {len(results)}/{len(examples)}...")
                
        except Exception as e:
            print(f"  Error on example {ex['id']}: {e}")
            continue
    
    conn.commit()
    return results,total_input_tokens, total_output_tokens


def print_summary(results,total_input_tokens, total_output_tokens):
    """Print evaluation summary."""
    total = len(results)
    noul_acc = sum(r["noul_correct"] for r in results) / total
    choice_acc = sum(r["choice_correct"] for r in results) / total
    avg_conf = sum(r["confidence"] for r in results) / total
    
    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total examples:        {total}")
    print(f"Noul accuracy:         {noul_acc:.2%}")
    print(f"Choice accuracy:       {choice_acc:.2%}")
    print(f"Average confidence:    {avg_conf:.3f}")
    print(f"{'='*50}\n")
    print(f"Total input tokens:  {total_input_tokens:,}")
    print(f"Total output tokens: {total_output_tokens:,}")
    print(f"Avg input per call:  {total_input_tokens/total:.0f}")
        
    # Cost calculation (official pricing: $0.042 per million input, output free)
    cost = (total_input_tokens / 1000000) * 0.042
    print(f"Estimated cost:      ${cost:.6f}")


def run_analysis(conn):
    """Run calibration and confidence analysis after evaluation."""
    conn.close()  # Close write connection before analysis
    
    print("Running calibration analysis...")
    cal = compute_calibration(DB_PATH)
    if cal["bin_centers"]:
        print(f"  Calibration bins: {len(cal['bin_centers'])}")
    else:
        print("  No calibration data.")
    
    print("Running confidence analysis...")
    conf = compute_confidence(DB_PATH)
    if conf["error"]:
        print(f"  Confidence analysis error: {conf['error']}")
    elif conf["bin_centers"]:
        print(f"  Confidence bins: {len(conf['bin_centers'])}")
        print(f"  Examples with confidence: {sum(conf['bin_counts'])}")
    else:
        print("  No confidence data.")
    
    return cal, conf


def main():
    print("Loading examples...")
    examples = load_examples()
    print(f"  Loaded {len(examples)} examples.")
    
    print("Setting up database...")
    conn = setup_database()
    
    print("Running evaluation...")
    results,total_input_tokens, total_output_tokens = run_evaluation(examples, conn)
    
    print_summary(results,total_input_tokens, total_output_tokens)
    
    # Run analysis (this closes the connection internally)
    run_analysis(conn)
    
    print("Done. Run 'streamlit run app.py' to view the dashboard.")


if __name__ == "__main__":
    main()
