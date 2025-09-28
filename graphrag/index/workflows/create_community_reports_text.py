# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import logging

import pandas as pd

import graphrag.data_model.schemas as schemas
from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.defaults import graphrag_config_defaults
from graphrag.config.enums import AsyncType
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.operations.finalize_community_reports import (
    finalize_community_reports,
)
from graphrag.index.operations.summarize_communities.explode_communities import (
    explode_communities,
)
from graphrag.index.operations.summarize_communities.summarize_communities import (
    summarize_communities,
)
from graphrag.index.operations.summarize_communities.text_unit_context.context_builder import (
    build_level_context,
    build_local_context,
)
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage

logger = logging.getLogger(__name__)


def _log_dataframe_info(name: str, df: pd.DataFrame | None) -> None:
    """Log basic details about a dataframe for debugging."""
    if df is None:
        logger.info("%s dataframe is None", name)
        return

    rows, cols = df.shape
    column_preview = list(df.columns[:6])
    if cols > 6:
        column_preview.append("...")
    logger.info(
        "%s dataframe shape=(%s, %s) columns=%s",
        name,
        rows,
        cols,
        column_preview,
    )


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to transform community reports."""
    logger.info("Workflow started: create_community_reports_text")
    entities = await load_table_from_storage("entities", context.output_storage)
    communities = await load_table_from_storage("communities", context.output_storage)

    text_units = await load_table_from_storage("text_units", context.output_storage)

    _log_dataframe_info("entities", entities)
    _log_dataframe_info("communities", communities)
    _log_dataframe_info("text_units", text_units)

    community_reports_llm_settings = config.get_language_model_config(
        config.community_reports.model_id
    )
    async_mode = community_reports_llm_settings.async_mode
    num_threads = community_reports_llm_settings.concurrent_requests
    summarization_strategy = config.community_reports.resolved_strategy(
        config.root_dir, community_reports_llm_settings
    )

    logger.info(
        "Community report summarization settings: model_id=%s async_mode=%s num_threads=%s",
        community_reports_llm_settings.model_id,
        getattr(async_mode, "value", async_mode),
        num_threads,
    )
    logger.debug(
        "Community report summarization strategy keys: %s",
        sorted(summarization_strategy.keys()),
    )

    output = await create_community_reports_text(
        entities,
        communities,
        text_units,
        context.callbacks,
        context.cache,
        summarization_strategy,
        async_mode=async_mode,
        num_threads=num_threads,
    )

    await write_table_to_storage(output, "community_reports", context.output_storage)

    logger.info("Workflow completed: create_community_reports_text")
    return WorkflowFunctionOutput(result=output)


async def create_community_reports_text(
    entities: pd.DataFrame,
    communities: pd.DataFrame,
    text_units: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    summarization_strategy: dict,
    async_mode: AsyncType = AsyncType.AsyncIO,
    num_threads: int = 4,
) -> pd.DataFrame:
    """All the steps to transform community reports."""
    nodes = explode_communities(communities, entities)
    _log_dataframe_info("exploded_nodes", nodes)

    community_count = (
        communities[schemas.COMMUNITY_ID].nunique()
        if schemas.COMMUNITY_ID in communities
        else len(communities)
    )
    level_count = (
        communities[schemas.COMMUNITY_LEVEL].nunique()
        if schemas.COMMUNITY_LEVEL in communities
        else "unknown"
    )
    logger.info(
        "Preparing to summarize %s communities across %s levels",
        community_count,
        level_count,
    )

    summarization_strategy["extraction_prompt"] = summarization_strategy["text_prompt"]

    max_input_length = summarization_strategy.get(
        "max_input_length", graphrag_config_defaults.community_reports.max_input_length
    )
    logger.info(
        "Community report context max_input_length=%s async_mode=%s num_threads=%s",
        max_input_length,
        getattr(async_mode, "value", async_mode),
        num_threads,
    )

    local_contexts = build_local_context(
        communities, text_units, nodes, max_input_length
    )
    exceed_flag = (
        int(local_contexts[schemas.CONTEXT_EXCEED_FLAG].sum())
        if schemas.CONTEXT_EXCEED_FLAG in local_contexts
        else 0
    )
    logger.info(
        "Built %s local context records (%s exceeding token limit)",
        len(local_contexts),
        exceed_flag,
    )
    logger.debug(
        "Local context columns: %s",
        list(local_contexts.columns),
    )

    community_reports = await summarize_communities(
        nodes,
        communities,
        local_contexts,
        build_level_context,
        callbacks,
        cache,
        summarization_strategy,
        max_input_length=max_input_length,
        async_mode=async_mode,
        num_threads=num_threads,
    )
    _log_dataframe_info("raw_community_reports", community_reports)

    finalized = finalize_community_reports(community_reports, communities)
    _log_dataframe_info("finalized_community_reports", finalized)

    return finalized
