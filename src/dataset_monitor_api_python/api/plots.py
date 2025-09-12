from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from ..models.analysis import AnalysisRequest
from ..services import analysis, plotting

router = APIRouter()


@router.post("/", response_model=Dict[str, Any], tags=["Plots"])
def generate_plot_endpoint(request: AnalysisRequest):
    """
    Generates a Vega-Lite JSON specification for a single plot operation.
    """
    if len(request.operations) != 1:
        raise HTTPException(
            status_code=400, detail="Plot requests must have exactly one operation."
        )
    operation = request.operations[0]

    try:
        file_path = analysis.get_parquet_file_path(
            request.dataset, request.variant, request.version
        )
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Parquet file not found")

        plot_spec = plotting.generate_plot_spec(file_path, operation)
        return plot_spec

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )
