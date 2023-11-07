from typing import Any, Dict
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy import (
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
    __tablename__ = "offers"

    index = Column(Integer, primary_key=True)
    OFFER = Column(Text)
    RETAILER = Column(Text)
    BRAND = Column(Text)
    CATEGORIES = Column(Text)
    SUPER_CATEGORIES = Column(Text)
    TARGET = Column(Text)


class OfferDBSession:
    def __init__(self, db_path="data/processed/database.sqlite") -> None:
        self.db_path = db_path
        self.session: Session | Any = None
        self.new_session()

    def __getattr__(self, attr: str) -> None:
        return getattr(self.session, attr)

    def __del__(self) -> None:
        self.close()

    def new_session(self) -> None:
        if not self.session:
            engine = create_engine(f"sqlite:///{self.db_path}")
            Base.metadata.create_all(engine)
            session = sessionmaker(bind=engine)
            self.session = session()

    def execute(
        self, query: str | TextClause, raw: bool = True
    ) -> list[Dict[str, Any] | Any] | Sequence | Any | Result[Any]:
        if isinstance(query, str):
            query = text(query)

        result = self.session.execute(query)

        if not raw:
            result = result.fetchall()

        return result

    def get_targets(self) -> list[str]:
        targets = (
            self.session.query(OffersTable.TARGET).order_by(OffersTable.index).all()
        )
        targets = [t[0] for t in targets]
        return targets

    def get_offers(self) -> list[str]:
        offers = self.session.query(OffersTable.OFFER).order_by(OffersTable.index).all()
        offers = [o[0] for o in offers]
        return offers

    def get_rows(
        self,
        indices: list[int],
        as_dict: bool = False,
    ) -> list[list[Any]]:
        result = (
            self.session.query(
                OffersTable.index,
                OffersTable.OFFER,
                OffersTable.RETAILER,
                OffersTable.BRAND,
                OffersTable.CATEGORIES,
                OffersTable.SUPER_CATEGORIES,
            )
            .filter(OffersTable.index.in_(indices))
            .all()
        )

        if as_dict:
            result = [row._asdict() for row in result]

        return result

    def close(self) -> None:
        self.session.close()
        self.session = None
