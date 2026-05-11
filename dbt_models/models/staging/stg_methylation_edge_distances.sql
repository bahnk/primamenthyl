{{ config(tags=['individual']) }}

with methylation_fragments as (
    select
        records.query_id,
        methylation.align_id,
        methylation.reference,
        methylation.context,
        methylation.methylated,
        methylation.methylation_offset,
        methylation.position,
        fragments.start_position,
        fragments.end_position,
        methylation.position - fragments.start_position as dist_left,
        fragments.end_position - methylation.position as dist_right
    from {{ ref('stg_methylation_positions') }} as methylation
    inner join {{ source('quant', 'quant__records') }} as records
        on methylation.align_id = records.align_id
    inner join {{ ref('stg_query_fragments') }} as fragments
        on records.query_id = fragments.query_id
)

select
    samples.sample,
    query_id,
    align_id,
    reference,
    context,
    methylated,
    methylation_offset,
    position,
    start_position,
    end_position,
    case
        when dist_left <= dist_right then -dist_left
        else dist_right
    end as edge_distance
from methylation_fragments
cross join {{ source('quant', 'quant__samples') }} as samples
