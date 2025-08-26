from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np

__all__ = ['CoaddPrimitive']


class CoaddPrimitive(BasePrimitive):
    """
    ARGUMENTS:
        tracepos (datamodels.HISPECTraceProsModel | None): Trace centers to do fitting on
        Deg (int | 4): Polynomial order to use for smoothing trace profile parameters

    """

    class_alias = 'coadd_frames'

    ARGUMENTS = {
        'suffix' : 'coadded',
    }

    def _perform(self):
        ### This is assuming self.args.name is a list, we need some way to pass
        ### an arbitrary number of L0 datamodels to this primitive
        input_models = [datamodels.open(name) for name in self.args.name]
        ### Do we want to include names of coadded files?
        coadded = coadd_frames(input_models)
        ### Not sure if this is what we should be returning
        return Arguments(name=coadded)




def coadd_frames(
        input_models : list,
    ) -> datamodels.HISPECL0Model:
    """
    Coadd a given sequence of HISPEC L0 data models

    Args:
        input_models (list): List of L0 data model objects

    Returns:
        HISPECL0Model: Model with the data extensions filled by coadding
    """
    ### Use the first model as a base

    for model in input_models:
        self.log.info(f"Coadding {self.args.name}")

    nframe = len(input_models)
    outdata =    np.zeros((nframe,4096,4096))
    outerr =     np.zeros((nframe,4096,4096))
    outrnoise =  np.zeros((nframe,4096,4096))
    outpoisson = np.zeros((nframe,4096,4096))
    outdq =      np.zeros((nframe,4096,4096))
    outmidtime = np.zeros((nframe,4096,4096))
    
    for i in range(len(input_models)):
        outdata[i] = input_models[i]['DATA']
        outerr[i] = input_models[i]['ERR']
        outrnoise[i] = input_models[i]['VAR_RNOISE']
        outpoisson[i] = input_models[i]['VAR_POISSON']
        outdq[i] = input_models[i]['DQ']
        outmidtime[i] = input_models[i]['MID_TIME']

    ### assume that errors sum in quadrature
    model_result = datamodels.HISPECL0Model(
         data=dict(
                data=np.sum(outdata,axis=0),
                err=np.sqrt(np.sum(outerr**2,axis=0)),
                var_poisson=np.sqrt(np.sum(outpoisson**2,axis=0)),
                var_rnoise=np.sqrt(np.sum(outrnoise**2,axis=0)),
                dq=np.sum(outdq), ### Not sure this is what we want to do if dq vals have meaning
                mid_time=np.mean(out_midtime,axis=0)
            ),
            ### Just use the first frames's metadata for now, might want to do something else
            meta=input_models[0].meta
        )

        return Arguments(name=model_result)