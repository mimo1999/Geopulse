"""Quick connectivity smoke-test against local gdelt_risk database."""
import psycopg2
import psycopg2.extras

DSN = "postgresql://gldt:gldt@localhost:5432/gdelt_risk"

conn = psycopg2.connect(DSN, cursor_factory=psycopg2.extras.RealDictCursor)
with conn.cursor() as cur:
    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
          AND table_name NOT LIKE 'gdelt_events_20%%'
        ORDER BY table_name
    """)
    tables = [r["table_name"] for r in cur.fetchall()]

    cur.execute("SELECT COUNT(*) AS n FROM information_schema.tables WHERE table_schema='public' AND table_name LIKE 'gdelt_events_20%%'")
    partitions = cur.fetchone()["n"]

conn.close()

print(f"Connected to {DSN}")
print(f"Core tables ({len(tables)}):")
for t in tables:
    print(f"  {t}")
print(f"gdelt_events partitions: {partitions}")
print("OK")
