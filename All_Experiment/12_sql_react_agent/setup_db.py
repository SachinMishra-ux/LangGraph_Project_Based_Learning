import sqlite3
import datetime
import random

def setup_database(db_path="sales.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table for customers
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        segment TEXT NOT NULL
    )
    """)

    # Create table for products
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL
    )
    """)

    # Create table for sales transactions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER,
        product_id INTEGER,
        quantity INTEGER,
        sale_date DATE,
        FOREIGN KEY(customer_id) REFERENCES customers(id),
        FOREIGN KEY(product_id) REFERENCES products(id)
    )
    """)

    # Clear existing data if any (for fresh start on reruns)
    cursor.execute("DELETE FROM transactions")
    cursor.execute("DELETE FROM products")
    cursor.execute("DELETE FROM customers")

    # Insert mock customers
    customers = [
        ("Alice Smith", "Retail"),
        ("Bob Johnson", "Wholesale"),
        ("Charlie Brown", "Retail"),
        ("Diana Prince", "Corporate")
    ]
    cursor.executemany("INSERT INTO customers (name, segment) VALUES (?, ?)", customers)

    # Insert mock products
    products = [
        ("Laptop Pro", "Electronics", 1200.00),
        ("Wireless Mouse", "Electronics", 25.50),
        ("Ergonomic Chair", "Furniture", 250.00),
        ("Standing Desk", "Furniture", 400.00)
    ]
    cursor.executemany("INSERT INTO products (name, category, price) VALUES (?, ?, ?)", products)

    # Ingest mock transactions spanning last 30 days
    today = datetime.date.today()
    for _ in range(50):
        c_id = random.randint(1, len(customers))
        p_id = random.randint(1, len(products))
        qty = random.randint(1, 5)
        # Random date within last 30 days
        days_ago = random.randint(0, 30)
        sale_date = today - datetime.timedelta(days=days_ago)
        
        cursor.execute(
            "INSERT INTO transactions (customer_id, product_id, quantity, sale_date) VALUES (?, ?, ?, ?)",
            (c_id, p_id, qty, sale_date.isoformat())
        )

    conn.commit()
    conn.close()
    print(f"Database setup complete at {db_path}. Ingested sample sales data.")

if __name__ == "__main__":
    setup_database()
