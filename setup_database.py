# setup_database.py
"""
Script to create and populate a SQLite database with sample company data
for testing SQL agents with CrewAI
"""

import sqlite3
import pandas as pd
import random
from datetime import datetime, timedelta
import os

def create_database(db_path='company_data.db'):
    """Create SQLite database with sample company data"""
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")
    
    # Create connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    print(f"Created new database: {db_path}")
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON")
    
    # Create tables with proper schema
    
    # 1. Departments table
    cursor.execute("""
    CREATE TABLE departments (
        department_id INTEGER PRIMARY KEY AUTOINCREMENT,
        department_name TEXT NOT NULL UNIQUE,
        manager_id INTEGER,
        budget DECIMAL(10, 2),
        location TEXT
    )
    """)
    print("Created departments table")
    
    # 2. Employees table
    cursor.execute("""
    CREATE TABLE employees (
        employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        phone TEXT,
        department_id INTEGER,
        job_title TEXT,
        hire_date DATE NOT NULL,
        birth_date DATE,
        address TEXT,
        city TEXT,
        country TEXT DEFAULT 'USA',
        reports_to INTEGER,
        FOREIGN KEY (department_id) REFERENCES departments(department_id),
        FOREIGN KEY (reports_to) REFERENCES employees(employee_id)
    )
    """)
    print("Created employees table")
    
    # 3. Salaries table
    cursor.execute("""
    CREATE TABLE salaries (
        salary_id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id INTEGER NOT NULL,
        salary DECIMAL(10, 2) NOT NULL,
        effective_date DATE NOT NULL,
        end_date DATE,
        currency TEXT DEFAULT 'USD',
        FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
    )
    """)
    print("Created salaries table")
    
    # 4. Sales table
    cursor.execute("""
    CREATE TABLE sales (
        sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id INTEGER NOT NULL,
        customer_name TEXT NOT NULL,
        product_name TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        unit_price DECIMAL(10, 2) NOT NULL,
        total_amount DECIMAL(10, 2) NOT NULL,
        sale_date DATE NOT NULL,
        region TEXT,
        FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
    )
    """)
    print("Created sales table")
    
    # 5. Projects table
    cursor.execute("""
    CREATE TABLE projects (
        project_id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_name TEXT NOT NULL,
        department_id INTEGER,
        start_date DATE NOT NULL,
        end_date DATE,
        budget DECIMAL(10, 2),
        status TEXT CHECK(status IN ('Planning', 'Active', 'Completed', 'On Hold')),
        project_manager_id INTEGER,
        FOREIGN KEY (department_id) REFERENCES departments(department_id),
        FOREIGN KEY (project_manager_id) REFERENCES employees(employee_id)
    )
    """)
    print("Created projects table")
    
    # 6. Performance Reviews table
    cursor.execute("""
    CREATE TABLE performance_reviews (
        review_id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id INTEGER NOT NULL,
        review_date DATE NOT NULL,
        reviewer_id INTEGER NOT NULL,
        rating INTEGER CHECK(rating >= 1 AND rating <= 5),
        comments TEXT,
        FOREIGN KEY (employee_id) REFERENCES employees(employee_id),
        FOREIGN KEY (reviewer_id) REFERENCES employees(employee_id)
    )
    """)
    print("Created performance_reviews table")
    
    # Insert sample data
    
    # Departments
    departments = [
        ('Sales', 120000, 'New York'),
        ('Engineering', 500000, 'San Francisco'),
        ('Marketing', 80000, 'Los Angeles'),
        ('HR', 60000, 'Chicago'),
        ('Finance', 100000, 'New York'),
        ('Customer Support', 70000, 'Austin'),
        ('Product', 150000, 'San Francisco'),
        ('Operations', 90000, 'Chicago')
    ]
    
    for dept in departments:
        cursor.execute("""
        INSERT INTO departments (department_name, budget, location) 
        VALUES (?, ?, ?)
        """, dept)
    print(f"Inserted {len(departments)} departments")
    
    # Sample employee data
    first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emma', 'Chris', 'Lisa', 
                   'Robert', 'Maria', 'James', 'Jennifer', 'William', 'Patricia', 'Richard', 
                   'Linda', 'Thomas', 'Barbara', 'Charles', 'Susan']
    
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 
                  'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 
                  'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin']
    
    job_titles = {
        'Sales': ['Sales Representative', 'Sales Manager', 'Account Executive', 'Sales Director'],
        'Engineering': ['Software Engineer', 'Senior Engineer', 'Tech Lead', 'Engineering Manager'],
        'Marketing': ['Marketing Specialist', 'Marketing Manager', 'Content Writer', 'SEO Specialist'],
        'HR': ['HR Specialist', 'HR Manager', 'Recruiter', 'HR Director'],
        'Finance': ['Financial Analyst', 'Senior Accountant', 'Finance Manager', 'CFO'],
        'Customer Support': ['Support Agent', 'Support Lead', 'Support Manager'],
        'Product': ['Product Manager', 'Senior PM', 'Product Director'],
        'Operations': ['Operations Analyst', 'Operations Manager', 'COO']
    }
    
    # Generate employees
    employee_id = 1
    employees_data = []
    
    for _ in range(100):  # Create 100 employees
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        email = f"{first_name.lower()}.{last_name.lower()}{random.randint(1,99)}@company.com"
        phone = f"+1-{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}"
        
        # Assign to department
        dept_id = random.randint(1, len(departments))
        dept_name = departments[dept_id-1][0]
        job_title = random.choice(job_titles[dept_name])
        
        # Generate dates
        hire_date = datetime.now() - timedelta(days=random.randint(30, 3650))  # 1 month to 10 years ago
        birth_year = datetime.now().year - random.randint(22, 65)  # 22-65 years old
        birth_date = datetime(birth_year, random.randint(1, 12), random.randint(1, 28))
        
        # Address
        cities = ['New York', 'San Francisco', 'Los Angeles', 'Chicago', 'Austin', 
                  'Seattle', 'Boston', 'Denver', 'Miami', 'Portland']
        city = random.choice(cities)
        address = f"{random.randint(1, 9999)} {random.choice(['Main', 'First', 'Second', 'Oak', 'Pine', 'Maple'])} Street"
        
        employees_data.append((
            first_name, last_name, email, phone, dept_id, job_title,
            hire_date.strftime('%Y-%m-%d'), birth_date.strftime('%Y-%m-%d'),
            address, city, 'USA'
        ))
        employee_id += 1
    
    # Insert employees
    cursor.executemany("""
    INSERT INTO employees (first_name, last_name, email, phone, department_id, 
                          job_title, hire_date, birth_date, address, city, country)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, employees_data)
    print(f"Inserted {len(employees_data)} employees")
    
    # Set some managers (reports_to)
    cursor.execute("""
    UPDATE employees 
    SET reports_to = (SELECT employee_id FROM employees WHERE job_title LIKE '%Manager%' 
                     AND department_id = employees.department_id LIMIT 1)
    WHERE job_title NOT LIKE '%Manager%' AND job_title NOT LIKE '%Director%'
    """)
    
    # Update department managers
    cursor.execute("""
    UPDATE departments 
    SET manager_id = (SELECT employee_id FROM employees 
                     WHERE department_id = departments.department_id 
                     AND (job_title LIKE '%Director%' OR job_title LIKE '%Manager%') 
                     LIMIT 1)
    """)
    
    # Generate salaries
    salary_data = []
    salary_ranges = {
        'Representative': (45000, 65000),
        'Specialist': (50000, 70000),
        'Analyst': (60000, 85000),
        'Engineer': (80000, 150000),
        'Senior': (90000, 160000),
        'Lead': (100000, 170000),
        'Manager': (90000, 150000),
        'Director': (120000, 200000),
        'CFO': (180000, 300000),
        'COO': (180000, 300000)
    }
    
    cursor.execute("SELECT employee_id, job_title, hire_date FROM employees")
    for emp_id, job_title, hire_date in cursor.fetchall():
        # Determine salary range based on job title
        base_salary = 55000  # default
        for key, (min_sal, max_sal) in salary_ranges.items():
            if key in job_title:
                base_salary = random.randint(min_sal, max_sal)
                break
        
        # Current salary (with some increases over time)
        current_salary = base_salary * (1 + random.uniform(0, 0.3))  # 0-30% increase
        
        salary_data.append((
            emp_id,
            round(current_salary, 2),
            hire_date,  # Salary effective from hire date
            None,  # No end date for current salary
            'USD'
        ))
    
    cursor.executemany("""
    INSERT INTO salaries (employee_id, salary, effective_date, end_date, currency)
    VALUES (?, ?, ?, ?, ?)
    """, salary_data)
    print(f"Inserted {len(salary_data)} salary records")
    
    # Generate sales data (for sales employees only)
    products = ['Software License', 'Cloud Storage', 'API Access', 'Support Package', 
                'Training Program', 'Consulting Service', 'Hardware Device', 'Premium Subscription']
    
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East']
    
    cursor.execute("""
    SELECT employee_id FROM employees 
    WHERE department_id = (SELECT department_id FROM departments WHERE department_name = 'Sales')
    """)
    sales_employees = [row[0] for row in cursor.fetchall()]
    
    sales_data = []
    for _ in range(500):  # Generate 500 sales records
        emp_id = random.choice(sales_employees)
        customer = f"{random.choice(['Tech', 'Global', 'Digital', 'Smart'])} {random.choice(['Corp', 'Inc', 'Solutions', 'Systems'])}"
        product = random.choice(products)
        quantity = random.randint(1, 50)
        unit_price = random.randint(100, 5000)
        total = quantity * unit_price
        sale_date = datetime.now() - timedelta(days=random.randint(1, 730))  # Last 2 years
        region = random.choice(regions)
        
        sales_data.append((
            emp_id, customer, product, quantity, unit_price, total,
            sale_date.strftime('%Y-%m-%d'), region
        ))
    
    cursor.executemany("""
    INSERT INTO sales (employee_id, customer_name, product_name, quantity, 
                      unit_price, total_amount, sale_date, region)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, sales_data)
    print(f"Inserted {len(sales_data)} sales records")
    
    # Generate projects
    project_names = ['Website Redesign', 'Mobile App Development', 'Data Migration', 
                     'CRM Implementation', 'Security Audit', 'Market Research', 
                     'Product Launch', 'Cost Reduction Initiative', 'AI Integration',
                     'Customer Portal', 'API Development', 'Training Program']
    
    projects_data = []
    for i, proj_name in enumerate(project_names):
        dept_id = random.randint(1, len(departments))
        start_date = datetime.now() - timedelta(days=random.randint(30, 500))
        end_date = start_date + timedelta(days=random.randint(30, 365))
        budget = random.randint(10000, 500000)
        status = random.choice(['Planning', 'Active', 'Completed', 'On Hold'])
        
        # Get a manager from the department
        cursor.execute("""
        SELECT employee_id FROM employees 
        WHERE department_id = ? AND job_title LIKE '%Manager%' 
        LIMIT 1
        """, (dept_id,))
        manager_result = cursor.fetchone()
        manager_id = manager_result[0] if manager_result else None
        
        projects_data.append((
            proj_name, dept_id, start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d') if status == 'Completed' else None,
            budget, status, manager_id
        ))
    
    cursor.executemany("""
    INSERT INTO projects (project_name, department_id, start_date, end_date, 
                         budget, status, project_manager_id)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, projects_data)
    print(f"Inserted {len(projects_data)} projects")
    
    # Generate performance reviews
    reviews_data = []
    cursor.execute("SELECT employee_id, hire_date FROM employees")
    employees_for_review = cursor.fetchall()
    
    for emp_id, hire_date in employees_for_review[:50]:  # Reviews for first 50 employees
        # Get potential reviewers (managers)
        cursor.execute("""
        SELECT employee_id FROM employees 
        WHERE job_title LIKE '%Manager%' AND employee_id != ?
        LIMIT 5
        """, (emp_id,))
        reviewers = [row[0] for row in cursor.fetchall()]
        
        if reviewers:
            review_date = datetime.now() - timedelta(days=random.randint(1, 180))
            reviewer_id = random.choice(reviewers)
            rating = random.randint(3, 5)  # Ratings from 3 to 5
            
            comments = [
                "Excellent performance, exceeds expectations",
                "Good work, meets all objectives",
                "Solid contributor to the team",
                "Shows great potential for growth",
                "Consistently delivers quality work"
            ]
            
            reviews_data.append((
                emp_id, review_date.strftime('%Y-%m-%d'),
                reviewer_id, rating, random.choice(comments)
            ))
    
    cursor.executemany("""
    INSERT INTO performance_reviews (employee_id, review_date, reviewer_id, rating, comments)
    VALUES (?, ?, ?, ?, ?)
    """, reviews_data)
    print(f"Inserted {len(reviews_data)} performance reviews")
    
    # Commit and close
    conn.commit()
    conn.close()
    
    print(f"\n Database setup complete: {db_path}")
    print("\nDatabase contains:")
    print(f"  - {len(departments)} departments")
    print(f"  - {len(employees_data)} employees")
    print(f"  - {len(salary_data)} salary records")
    print(f"  - {len(sales_data)} sales transactions")
    print(f"  - {len(projects_data)} projects")
    print(f"  - {len(reviews_data)} performance reviews")
    
    return db_path

def verify_database(db_path='company_data.db'):
    """Verify the database was created correctly"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\nDatabase Verification:")
    print("-" * 50)
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"{table_name}: {count} records")
    
    # Sample queries to verify relationships
    print("\n🔍 Sample Queries:")
    print("-" * 50)
    
    # Average salary by department
    cursor.execute("""
    SELECT d.department_name, AVG(s.salary) as avg_salary
    FROM departments d
    JOIN employees e ON d.department_id = e.department_id
    JOIN salaries s ON e.employee_id = s.employee_id
    WHERE s.end_date IS NULL
    GROUP BY d.department_name
    ORDER BY avg_salary DESC
    LIMIT 5
    """)
    
    print("\nAverage Salary by Department (Top 5):")
    for row in cursor.fetchall():
        print(f"  {row[0]}: ${row[1]:,.2f}")
    
    # Top sales performers
    cursor.execute("""
    SELECT e.first_name || ' ' || e.last_name as name, 
           SUM(s.total_amount) as total_sales
    FROM employees e
    JOIN sales s ON e.employee_id = s.employee_id
    GROUP BY e.employee_id
    ORDER BY total_sales DESC
    LIMIT 5
    """)
    
    print("\nTop 5 Sales Performers:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: ${row[1]:,.2f}")
    
    conn.close()

if __name__ == "__main__":
    # Create the database
    db_path = create_database()
    
    # Verify it was created correctly
    verify_database(db_path)
    
    print("\n Your database is ready to use with the SQL agents!")
    print(f"Database location: {os.path.abspath(db_path)}")