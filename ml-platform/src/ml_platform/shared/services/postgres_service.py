import os
from typing import Any
import dotenv
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from ml_platform.shared.paths import ENV_FILE
from ml_platform.shared.utils import get_healthy_endpoint

class PostgresService:
    def __init__(self, postgres_url: str | None = None, db_name: str | None = None):
        dotenv.load_dotenv(ENV_FILE)

        urls = [
            f"postgresql://{os.environ["POSTGRES_USER"]}:{os.environ["POSTGRES_PASSWORD"]}@postgres:5432/{db_name}",
            f"postgresql://{os.environ["POSTGRES_USER"]}:{os.environ["POSTGRES_PASSWORD"]}@0.0.0.0:5432/{db_name}"
        ]
        if postgres_url: urls.append(postgres_url + f"/{db_name}")

        self.database_url = get_healthy_endpoint(
            urls,
            health_path=None, command=["pg_isready", "-d"],
            timeout=0.5
        )

        self.pool = ConnectionPool(
            conninfo=self.database_url,
            min_size=1,
            max_size=10,
            kwargs={"row_factory": dict_row},
        )

    def close(self) -> None:
        self.pool.close()

    def execute( self, query: str, params: tuple[Any, ...] = (), ) -> None:
        with self.pool.connection() as conn:
            conn.execute(query, params)
            conn.commit()

    def fetch_one( self, query: str, params: tuple[Any, ...] = (), ) -> dict[str, Any] | None:
        with self.pool.connection() as conn:
            return conn.execute(query, params).fetchone()

    def fetch_all( self, query: str, params: tuple[Any, ...] = (), ) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            return conn.execute(query, params).fetchall()

    def insert( self, table: str, data: dict[str, Any], ) -> dict[str, Any] | None:
        columns = list(data.keys())
        values = list(data.values())

        column_sql = ", ".join(f'"{column}"' for column in columns)
        placeholders = ", ".join(["%s"] * len(values))

        query = f"""
            INSERT INTO "{table}" ({column_sql})
            VALUES ({placeholders})
            RETURNING *
        """

        with self.pool.connection() as conn:
            row = conn.execute(query, values).fetchone()
            conn.commit()
            return row


if __name__=="__main__":
    dotenv.load_dotenv(ENV_FILE)
    user = os.environ["POSTGRES_USER"]
    password= os.environ["POSTGRES_PASSWORD"]

    ps = PostgresService(postgres_url=f"postgresql://{user}:{password}@0.0.0.0:5432/", db_name="inference_server_db")
    r = ps.execute("""
    CREATE TABLE IF NOT EXISTS jobs ( id SERIAL PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL )
        """)

    print(r)