from typing import Any, Dict
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy import (
    Row,
    create_engine,
    text,
    Column,
    Integer,
    Text,
    TextClause,
    Result,
    Sequence,
)


Base = declarative_base()


class OffersTable(Base):
    """
    Represents the "offers" table in the database.

    Columns:
    - index (Integer): Primary key.
    - OFFER (Text): Offer text.
    - RETAILER (Text): Retailer name.
    - BRAND (Text): Brand name.
    - CATEGORIES (Text): Categories.
    - SUPER_CATEGORIES (Text): Super categories.
    - TARGET (Text): Target information.
    """

    __tablename__ = "offers"

    index = Column(Integer, primary_key=True)
    OFFER = Column(Text)
    RETAILER = Column(Text)
    BRAND = Column(Text)
    CATEGORIES = Column(Text)
    SUPER_CATEGORIES = Column(Text)
    TARGET = Column(Text)


class OfferDBSession:
    """
    A session class for interacting with the database containing offer data.

    Args:
    - db_path (str, optional): Path to the database file. Defaults to "data/processed/database.sqlite".

    Attributes:
    - db_path (str): Path to the database file.
    - session (Session | Any): SQLAlchemy database session.

    Methods:
    - new_session(self): Create a new database session.
    - execute(self, query: str | TextClause, raw: bool = True): Execute a SQL query and return results.
    - get_targets(self): Retrieve a list of target information from the database.
    - get_offers(self): Retrieve a list of offer texts from the database.
    - get_rows(self, indices: list[int] | None = None, as_dict: bool = False): Retrieve rows from the database based on indices.
    - close(self): Close the database session.
    """

    def __init__(self, db_path="data/processed/database.sqlite") -> None:
        """
        Initialize an OfferDBSession instance.

        Args:
        - db_path (str, optional): Path to the database file. Defaults to "data/processed/database.sqlite".
        """
        self.db_path = db_path
        self.session: Session | Any = None
        self.new_session()

    def __getattr__(self, attr: str) -> None:
        return getattr(self.session, attr)

    def __del__(self) -> None:
        self.close()

    def new_session(self) -> None:
        """
        Create a new database session if it doesn't exist.
        """
        if not self.session:
            engine = create_engine(f"sqlite:///{self.db_path}")
            Base.metadata.create_all(engine)
            session = sessionmaker(bind=engine)
            self.session = session()

    def execute(
        self, query: str | TextClause, raw: bool = True
    ) -> Sequence[Row[Any]] | Any | Result[Any]:
        """
        Execute a SQL query and return the results.

        Args:
        - query (str | TextClause): SQL query to execute.
        - raw (bool, optional): If True, return raw results as a list. If False, return a processed result set. Defaults to True.

        Returns:
        list[Dict[str, Any] | Any] | Sequence | Any | Result[Any]: Query results.
        """
        if isinstance(query, str):
            query = text(query)

        result = self.session.execute(query)

        if not raw:
            result = result.fetchall()

        return result

    def get_targets(self) -> list[str]:
        """
        Retrieve a list of target information from the database.

        Returns:
        list[str]: List of target information.
        """
        targets = (
            self.session.query(OffersTable.TARGET).order_by(OffersTable.index).all()
        )
        targets = [t[0] for t in targets]
        return targets

    def get_offers(self) -> list[str]:
        """
        Retrieve a list of offer texts from the database.

        Returns:
        list[str]: List of offer texts.
        """
        offers = self.session.query(OffersTable.OFFER).order_by(OffersTable.index).all()
        offers = [o[0] for o in offers]
        return offers

    def get_rows(
        self,
        indices: list[int] | None = None,
        as_dict: bool = False,
    ) -> list[list[Any]]:
        """
        Retrieve rows from the database based on indices.

        Args:
        - indices (list[int] | None, optional): List of indices to retrieve. If None, retrieve all rows. Defaults to None.
        - as_dict (bool, optional): If True, return rows as dictionaries. If False, return rows as lists. Defaults to False.

        Returns:
        list[list[Any]]: List of rows from the database.
        """

        query = self.session.query(
            OffersTable.index,
            OffersTable.OFFER,
            OffersTable.RETAILER,
            OffersTable.BRAND,
            OffersTable.CATEGORIES,
            OffersTable.SUPER_CATEGORIES,
        )
        if indices:
            query = query.filter(OffersTable.index.in_(indices))

        result = query.all()

        if as_dict:
            result = [row._asdict() for row in result]

        return result

    def close(self) -> None:
        """
        Close the database session.
        """
        self.session.close()
        self.session = None
