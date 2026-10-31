use crate::{Backend, Host, TensorEntry, compute, guest_error, invalid_handle};
use forja_core::{CommandList, DType, Layout, Op, Tensor};
use wasmtime::component::Resource;

impl<B: Backend + 'static> Host<B> {
    pub(crate) fn prepare_selection_logits(
        &self,
        pool: u32,
        logits: &Resource<TensorEntry>,
        bucket: u32,
        active: impl Iterator<Item = usize>,
    ) -> Result<Option<crate::SubmitRequest<B>>, compute::Error> {
        let Some((_, slot, _)) = self.input_slot(pool)? else {
            return Ok(None);
        };
        let Some(inputs) = &slot.inputs else {
            return Ok(None);
        };
        let vocab = slot
            .input_layout
            .ok_or_else(|| invalid_handle("state input layout is missing"))?
            .vocab;
        let count = usize::try_from(bucket).map_err(invalid_handle)?;
        let snapshot = self.reserve_references(
            count
                .checked_add(2)
                .ok_or_else(|| invalid_handle("selection reference count overflowed"))?,
        )?;
        let active = active
            .take(
                count
                    .checked_add(1)
                    .ok_or_else(|| invalid_handle("selection lane count overflowed"))?,
            )
            .collect::<Vec<_>>();
        if count > slot.binding.lanes.len()
            || active.is_empty()
            || active.len() > count
            || active.iter().any(|&lane| lane >= count)
            || active.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid_handle("selection lane geometry is invalid"));
        }
        let source = self.entry(logits)?.clone();
        self.taints
            .check(source.tensor.buffer())
            .map_err(guest_error)?;
        let rows = match source.tensor.layout().shape() {
            [width] if *width == vocab => 1,
            [rows, width] if *width == vocab => *rows,
            _ => {
                return Err(invalid_handle(
                    "selection logits have incompatible dimensions",
                ));
            }
        };
        if source.symbolic.is_some()
            || source.tensor.layout().dtype() != DType::F32
            || (rows != bucket && !(rows == 1 && active.len() == 1))
        {
            return Err(invalid_handle(
                "selection logits require a matching batch or one active singleton",
            ));
        }
        let handle = inputs.banks[slot.active_bank].logits;
        let destination = self.entry(&Resource::new_borrow(handle))?.clone();
        let shared = source.tensor.buffer() == destination.tensor.buffer();
        let mut commands = CommandList::new();
        for lane in active.iter().copied() {
            let lane = u32::try_from(lane).map_err(invalid_handle)?;
            let source = logit_row(
                self.backend.as_ref(),
                &source.tensor,
                if rows == 1 { 0 } else { lane },
                vocab,
            )?
            .read_only();
            let destination = logit_row(self.backend.as_ref(), &destination.tensor, lane, vocab)?;
            if source.buffer() == destination.buffer() && source.layout() == destination.layout() {
                continue;
            }
            if shared && active.len() > 1 {
                return Err(invalid_handle(
                    "aliased selection logits require identical rows",
                ));
            }
            commands
                .dispatch(Op::Copy, &[&source], &destination)
                .map_err(guest_error)?;
        }
        if commands.dispatches().is_empty() {
            return Ok(None);
        }
        if !shared {
            self.recover_state_input_handles([handle])?;
        }
        let retained = vec![source, destination];
        let access = crate::SubmissionAccess::from_commands(&commands, &retained);
        let mut request = self.submission_request(
            crate::SubmissionWork::Commands(commands),
            access,
            retained,
            Vec::new(),
            None,
        );
        request.snapshot = Some(snapshot);
        Ok(Some(request))
    }
}

fn logit_row<B: Backend>(
    backend: &B,
    tensor: &Tensor,
    row: u32,
    vocab: u32,
) -> Result<Tensor, compute::Error> {
    let layout = tensor.layout();
    let (stride, column) = match layout.strides() {
        [column] => (0, *column),
        [stride, column] => (*stride, *column),
        _ => return Err(invalid_handle("selection logits require rank one or two")),
    };
    let offset = stride
        .checked_mul(u64::from(row))
        .and_then(|delta| layout.offset().checked_add(delta))
        .ok_or_else(|| invalid_handle("selection logit offset overflowed"))?;
    let layout = Layout::new(
        DType::F32,
        offset,
        vec![vocab],
        vec![column],
        tensor.buffer().byte_len(),
    )
    .map_err(guest_error)?;
    backend.tensor(tensor.buffer(), layout).map_err(guest_error)
}

impl<B: Backend + Send + Sync + 'static> Host<B> {
    /// Copies supplied logits into their physical selection rows without modifying the source.
    ///
    /// # Errors
    /// Refuses incompatible geometry, ambiguous aliases, failed sources or exhausted quota.
    pub async fn stage_selection_logits(
        &mut self,
        pool: &Resource<crate::PagePoolEntry>,
        logits: &Resource<TensorEntry>,
        bucket: u32,
        active: impl Iterator<Item = usize> + Send,
    ) -> Result<(), compute::Error> {
        if let Some(copy) = self.prepare_selection_logits(pool.rep(), logits, bucket, active)? {
            copy.run().await?;
        }
        Ok(())
    }
}
