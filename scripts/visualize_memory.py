# scripts/visualize_memory.py

import os
import chromadb

def main():
    # Force absolute path relative to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, "data", "chroma_db")
    
    if not os.path.exists(db_path):
        print(f"Database folder not found at {db_path}. Has the Sleep Cycle run yet?")
        return

    print(f"Loading ChromaDB Memory from: {db_path}\n")
    client = chromadb.PersistentClient(path=db_path)
    
    target_collections = ["forced_zero_strategies", "rational_strategies"]
    markdown_output = "# Skull King AI: Discovered Strategies\n\n"

    for col_name in target_collections:
        try:
            collection = client.get_collection(name=col_name)
        except Exception:
            print(f"Collection '{col_name}' does not exist yet. Skipping.")
            continue
            
        # Explicitly ask for documents and metadata
        data = collection.get(include=["documents", "metadatas"])
        
        ids = data.get("ids",[])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas",[])
        
        header = f"## Persona: {col_name.upper().replace('_', ' ')}"
        print(f"{'='*60}\n{header}\n{'='*60}")
        markdown_output += f"{header}\n\n"
        
        if not ids:
            print("  (No rules memorized yet.)\n")
            markdown_output += "*No rules memorized yet.*\n\n"
            continue
            
        for i in range(len(ids)):
            rule_id = ids[i]
            rule_text = documents[i]
            meta = metadatas[i]
            
            # Console Output
            print(f"[{rule_id}]")
            print(f"Context : Round {meta.get('round_num', '?')} | Phase: {meta.get('phase', '?')} | Target Bid: {meta.get('bid', '?')}")
            print(f"Rule    : {rule_text}\n")
            
            # Markdown Output
            markdown_output += f"### {rule_id}\n"
            markdown_output += f"- **Context:** Round {meta.get('round_num', '?')} | Target Bid: {meta.get('bid', '?')}\n"
            markdown_output += f"- **Strategy:** {rule_text}\n\n"

    # Export to Markdown
    export_dir = os.path.join(base_dir, "data", "artifacts")
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, "Rules_Found.md")
    
    with open(export_path, "w", encoding="utf-8") as f:
        f.write(markdown_output)
        
    print(f"{'='*60}")
    print(f"Successfully exported all rules to: {export_path}")

if __name__ == "__main__":
    main()